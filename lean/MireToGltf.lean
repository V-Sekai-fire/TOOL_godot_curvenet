import LeanGltf

/-!
# `mire_to_gltf` — package a Mire-style triangle mesh as a `.glb`

Reads a positions buffer (raw `float32[3 × n_verts]`, little-endian)
and an indices buffer (raw `int32[3 × n_tris]`, little-endian) and
emits a glTF 2.0 binary file with one mesh + one node + one scene.

The output document includes an `EXT_structural_metadata` block
declaring the curvenet schema (Curvenet, Segment, Knot classes) so
downstream tools have a place to attach the rig data once it's
authored. This MVP doesn't fill in any property tables — the
schema is forward-compat scaffolding.

Usage:

    misc/mire_h_to_bin.py tests/mire_body_70k_data.h /tmp/mire
    cd lean && lake exe mire_to_gltf /tmp/mire/positions.f32 \
                                     /tmp/mire/tris.i32 \
                                     /tmp/mire/mire.glb
-/

open LeanGltf
open LeanGltf.Extensions.EXTStructuralMetadata

/-- Curvenet schema as `EXT_structural_metadata` classes. Mirrors the
    DeGoes22 §3 vocabulary (curve / segment / knot / sample / frame). -/
def curvenetSchema : Schema :=
  let f3 := { type := "SCALAR", componentType := some "FLOAT32", array := true, count := some 3 : ClassProperty }
  let f4 := { type := "SCALAR", componentType := some "FLOAT32", array := true, count := some 4 : ClassProperty }
  let f9 := { type := "SCALAR", componentType := some "FLOAT32", array := true, count := some 9 : ClassProperty }
  let nat := { type := "SCALAR", componentType := some "UINT32" : ClassProperty }
  let str := { type := "STRING" : ClassProperty }
  { id := some "curvenet"
  , name := some "Curvenet rig"
  , description := some "DeGoes22 §3 curvenet vocabulary plus DDM weights."
  , version := some "0.1.0"
  , classes := #[
      ("Curvenet", {
        name := some "Curvenet"
        description := some "A single profile-curve rig."
        properties := #[
          ("name",          { str  with description := some "Display name." }),
          ("segment_count", { nat  with description := some "Number of segments in this curvenet." }),
          ("sample_count",  { nat  with description := some "Total sample count along the curve." })
        ]
      }),
      ("Segment", {
        name := some "Segment"
        description := some "One cubic Bézier segment between two knots."
        properties := #[
          ("p0",          { f3 with description := some "Bézier control point 0." }),
          ("p1",          { f3 with description := some "Bézier control point 1." }),
          ("p2",          { f3 with description := some "Bézier control point 2." }),
          ("p3",          { f3 with description := some "Bézier control point 3." }),
          ("knot_kind_a", { nat with description := some "Start-knot kind (0 anchor / 1 regular / 2 intersection)." }),
          ("knot_kind_b", { nat with description := some "End-knot kind." })
        ]
      }),
      ("Knot", {
        name := some "Knot"
        description := some "Curvenet vertex (knot)."
        properties := #[
          ("position", { f3 with description := some "World-space position." }),
          ("kind",     { nat with description := some "0 anchor / 1 regular / 2 intersection." })
        ]
      }),
      ("Sample", {
        name := some "Sample"
        description := some "DeGoes22 §3 sample along a segment."
        properties := #[
          ("position",  { f3 with description := some "Sampled position." }),
          ("frame3x3",  { f9 with description := some "Row-major scaled-frame B·S as 9 floats." }),
          ("width_lhs", { f3 with description := some "Per-side width left." }),
          ("width_rhs", { f3 with description := some "Per-side width right." })
        ]
      }),
      ("DDMWeights", {
        name := some "Direct Delta Mush weight row"
        description := some "Per-vertex sparse weight row in CSR-like form."
        properties := #[
          ("handle_indices", { nat with description := some "Flat handle-index array (CSR col_idx)." }),
          ("weights",        { f4  with description := some "Flat weight array (CSR values)." }),
          ("row_offset",     { nat with description := some "Start offset into the flat arrays." }),
          ("row_length",     { nat with description := some "Number of non-zero entries for this vertex." })
        ]
      })
    ]
  }

/-- Read a binary file and pad it up to a 4-byte boundary so it can
    be concatenated with other buffer views without offset drift. -/
def readPadded4 (path : System.FilePath) : IO ByteArray := do
  let bytes ← IO.FS.readBinFile path
  let r := bytes.size % 4
  if r = 0 then return bytes
  let need := 4 - r
  let mut out := bytes
  for _ in [:need] do out := out.push 0
  return out

/-- Decode an IEEE-754 binary32 bit pattern into a `Float` (fp64).
    Lean's `Float.ofBits` takes fp64 bits, so for fp32 we hand-decode
    sign / exponent / mantissa and multiply by `2^exp`. NaN / Inf
    collapse to 0 so they don't poison min/max. -/
def fp32BitsToFloat (bits : UInt32) : Float :=
  let signBit : UInt32 := bits >>> 31
  let expBits : UInt32 := (bits >>> 23) &&& 0xFF
  let mantBits : UInt32 := bits &&& 0x7FFFFF
  let sign : Float := if signBit = 0 then 1.0 else -1.0
  if expBits == 0xFF then 0.0
  else if expBits == 0 then
    -- Subnormal: 2^{-126} · mant/2^23 = mant · 2^{-149}.
    sign * mantBits.toNat.toFloat * Float.pow 2.0 (-149.0)
  else
    let exp : Float := expBits.toNat.toFloat - 127.0
    let mant : Float := 1.0 + mantBits.toNat.toFloat * Float.pow 2.0 (-23.0)
    sign * mant * Float.pow 2.0 exp

/-- Read a little-endian fp32 at `base..base+3` of a `ByteArray`. -/
def readFp32LE (xs : ByteArray) (base : Nat) : Float :=
  let b0 := xs[base]!.toUInt32
  let b1 := xs[base+1]!.toUInt32
  let b2 := xs[base+2]!.toUInt32
  let b3 := xs[base+3]!.toUInt32
  let bits : UInt32 := b0 ||| (b1 <<< 8) ||| (b2 <<< 16) ||| (b3 <<< 24)
  fp32BitsToFloat bits

/-- Min/max accessor stats over a flat float buffer with 3-component
    stride. Returns 3-vec min, 3-vec max. -/
def vec3MinMax (xs : ByteArray) : Array Float × Array Float := Id.run do
  let n := xs.size / 12
  if n = 0 then return (#[0.0, 0.0, 0.0], #[0.0, 0.0, 0.0])
  let mut mn0 := readFp32LE xs 0;  let mut mx0 := mn0
  let mut mn1 := readFp32LE xs 4;  let mut mx1 := mn1
  let mut mn2 := readFp32LE xs 8;  let mut mx2 := mn2
  for i in [1:n] do
    let v0 := readFp32LE xs (12*i + 0)
    let v1 := readFp32LE xs (12*i + 4)
    let v2 := readFp32LE xs (12*i + 8)
    if v0 < mn0 then mn0 := v0
    if v0 > mx0 then mx0 := v0
    if v1 < mn1 then mn1 := v1
    if v1 > mx1 then mx1 := v1
    if v2 < mn2 then mn2 := v2
    if v2 > mx2 then mx2 := v2
  return (#[mn0, mn1, mn2], #[mx0, mx1, mx2])

def main (args : List String) : IO UInt32 := do
  match args with
  | positionsPath :: trisPath :: outPath :: _ => do
    let positions ← readPadded4 positionsPath
    let tris      ← readPadded4 trisPath
    let nVerts := positions.size / 12
    let nTris  := tris.size / 12
    if nVerts = 0 then
      IO.eprintln "no vertices in positions buffer"
      return 1
    let (pmin, pmax) := vec3MinMax positions

    -- Layout: [ positions | tris ]
    let posOffset : Nat := 0
    let triOffset : Nat := positions.size
    let bin : ByteArray := positions ++ tris

    let doc : Document := {
      asset := { generator := some "Curvenet / mire_to_gltf" },
      buffers := #[{ byteLength := bin.size }],
      bufferViews := #[
        { buffer := 0, byteOffset := posOffset, byteLength := positions.size,
          target := some BufferView.arrayBuffer },
        { buffer := 0, byteOffset := triOffset, byteLength := tris.size,
          target := some BufferView.elementArrayBuffer }
      ],
      accessors := #[
        { bufferView := some 0, componentType := 5126, count := nVerts,
          type := .vec3, min := some pmin, max := some pmax },
        { bufferView := some 1, componentType := 5125, count := nTris * 3,
          type := .scalar }
      ],
      meshes := #[{
        primitives := #[{
          attributes := #[("POSITION", 0)],
          indices := some 1,
          material := some 0
        }]
      }],
      materials := #[{ name := some "default" }],
      nodes := #[{ name := some "Mire", mesh := some 0 }],
      scenes := #[{ nodes := #[0], name := some "default" }],
      scene := some 0,
      extensions := #[
        (Top.extensionName, Top.toJson { schema := some curvenetSchema })
      ],
      extensionsUsed := #[Top.extensionName]
    }
    let glb := LeanGltf.GLB.emit doc bin
    IO.FS.writeBinFile outPath glb
    IO.println s!"wrote {outPath} ({glb.size} bytes; {nVerts} verts, {nTris} tris)"
    return 0
  | _ => do
    IO.eprintln "usage: mire_to_gltf <positions.f32> <tris.i32> <out.glb>"
    return 1
