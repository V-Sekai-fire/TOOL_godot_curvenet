import LeanGltf
import Curvenet.EndToEndExample

/-!
# `profile_curves_demo` — viewable `.glb` of the smallest Profile Curves rig

```
lake exe profile_curves_demo <out.glb>
```

Takes the rest + deformed positions computed by
`Curvenet.EndToEndExample` (a single cubic Bézier driving the
`triangleWithSample` 3-vertex mesh, pure-translation case) and writes a
glTF 2.0 binary with one mesh + one **morph-target animation** that
interpolates from rest → deformed over 1 second. Loads into any glTF
viewer (Blender 5.x, https://gltf-viewer.donmccurdy.com, macOS Quick
Look) and animates immediately.

The morph-target route was chosen so we *visibly verify the deformed
positions the solver produced*, not just a rigid transform applied to
the rest mesh. Even though this PR's specific test case (pure
translation) collapses to a rigid motion, the same demo skeleton
generalizes to rotations, scales, and any future curve-driven
deformation without changing the .glb structure.
-/

open LeanGltf
open Curvenet

/-! ## Little-endian fp32 / uint32 byte writers (no decode round-trip needed) -/

private def fp32EncodeBits (f : Float) : UInt32 :=
  let bits64 : UInt64 := f.toBits
  let sign64 : UInt64 := bits64 >>> 63
  let exp64  : UInt64 := (bits64 >>> 52) &&& 0x7FF
  let mant64 : UInt64 := bits64 &&& 0xFFFFFFFFFFFFF
  let sign32 : UInt32 := sign64.toUInt32 <<< 31
  if exp64 == 0x7FF then
    if mant64 ≠ 0 then sign32 ||| 0x7FC00000
    else sign32 ||| 0x7F800000
  else if exp64 == 0 then sign32
  else
    let unbiased : Int := (exp64.toNat : Int) - 1023
    let biased   : Int := unbiased + 127
    if biased ≤ 0 then sign32
    else if biased ≥ 0xFF then sign32 ||| 0x7F800000
    else
      let exp32  : UInt32 := biased.toNat.toUInt32 <<< 23
      let mant32 : UInt32 := (mant64 >>> 29).toUInt32 &&& 0x7FFFFF
      sign32 ||| exp32 ||| mant32

private def pushFp32LE (out : ByteArray) (f : Float) : ByteArray :=
  let bits := fp32EncodeBits f
  out.push  (bits &&& 0xFF).toUInt8
     |>.push ((bits >>> 8)  &&& 0xFF).toUInt8
     |>.push ((bits >>> 16) &&& 0xFF).toUInt8
     |>.push ((bits >>> 24) &&& 0xFF).toUInt8

private def pushU32LE (out : ByteArray) (n : UInt32) : ByteArray :=
  out.push  (n &&& 0xFF).toUInt8
     |>.push ((n >>> 8)  &&& 0xFF).toUInt8
     |>.push ((n >>> 16) &&& 0xFF).toUInt8
     |>.push ((n >>> 24) &&& 0xFF).toUInt8

private def pushVec3 (out : ByteArray) (v : Vec3) : ByteArray :=
  out |> (pushFp32LE · v.x) |> (pushFp32LE · v.y) |> (pushFp32LE · v.z)

/-! ## min / max over a vec3 array (POSITION accessor metadata requirement) -/

private def vec3Range (vs : Array Vec3) : Array Float × Array Float := Id.run do
  if vs.isEmpty then return (#[0.0, 0.0, 0.0], #[0.0, 0.0, 0.0])
  let v0 := vs[0]!
  let mut mn0 := v0.x;  let mut mn1 := v0.y;  let mut mn2 := v0.z
  let mut mx0 := mn0;   let mut mx1 := mn1;   let mut mx2 := mn2
  for v in vs do
    if v.x < mn0 then mn0 := v.x
    if v.x > mx0 then mx0 := v.x
    if v.y < mn1 then mn1 := v.y
    if v.y > mx1 then mx1 := v.y
    if v.z < mn2 then mn2 := v.z
    if v.z > mx2 then mx2 := v.z
  return (#[mn0, mn1, mn2], #[mx0, mx1, mx2])

/-! ## Main -/

def main (args : List String) : IO UInt32 := do
  let outPath := match args with
    | a :: _ => a
    | []     => "/tmp/profile_curves_demo.glb"

  let rest     := EndToEndExample.restPositions
  let deformed := EndToEndExample.deformedPositionsRotateZ90
  let indices  := EndToEndExample.indices

  -- POSITION delta = deformed − rest, per vertex. For the rotation case,
  -- each vertex's delta is different (unlike pure translation where they
  -- collapse to one shared offset), so the morph-target animation
  -- *visibly* deforms the triangle instead of looking like a rigid slide.
  let deltas : Array Vec3 := Array.ofFn (n := rest.size) (fun i =>
    let r := rest[i.val]!
    let d := deformed[i.val]!
    ⟨d.x - r.x, d.y - r.y, d.z - r.z⟩)

  /- Bin layout, all aligned to 4-byte boundaries:
       offset 0   : POSITION    (3 vec3 fp32 = 36 bytes)
       offset 36  : POSITION_d  (3 vec3 fp32 = 36 bytes)  -- morph delta
       offset 72  : INDICES     (3 uint32 = 12 bytes)
       offset 84  : TIME        (3 fp32 = 12 bytes)       -- 0.0, 1.0, 2.0
       offset 96  : WEIGHTS     (3 fp32 = 12 bytes)       -- 0.0, 1.0, 0.0  (ping-pong)
       offset 108 : end -/
  let mut bin : ByteArray := ByteArray.empty
  for v in rest     do bin := pushVec3  bin v
  let dOff : Nat := bin.size
  for v in deltas   do bin := pushVec3  bin v
  let iOff : Nat := bin.size
  for i in indices  do bin := pushU32LE bin i
  let tOff : Nat := bin.size
  bin := pushFp32LE bin 0.0
  bin := pushFp32LE bin 1.0
  bin := pushFp32LE bin 2.0
  let wOff : Nat := bin.size
  bin := pushFp32LE bin 0.0  -- rest
  bin := pushFp32LE bin 1.0  -- fully deformed
  bin := pushFp32LE bin 0.0  -- back to rest (ping-pong over 2 s; viewer loops the action)

  let nVerts := rest.size
  let nTris  := indices.size / 3
  let (pmin, pmax) := vec3Range rest
  let (dmin, dmax) := vec3Range deltas

  let bufferViews : Array BufferView := #[
    { buffer := 0, byteOffset := 0,    byteLength := dOff,             target := some BufferView.arrayBuffer },
    { buffer := 0, byteOffset := dOff, byteLength := iOff - dOff,      target := some BufferView.arrayBuffer },
    { buffer := 0, byteOffset := iOff, byteLength := tOff - iOff,      target := some BufferView.elementArrayBuffer },
    { buffer := 0, byteOffset := tOff, byteLength := wOff - tOff,      target := none },
    { buffer := 0, byteOffset := wOff, byteLength := bin.size - wOff,  target := none }
  ]
  let accessors : Array Accessor := #[
    { bufferView := some 0, componentType := 5126, count := nVerts, type := .vec3,
      min := some pmin, max := some pmax },                                         -- 0: POSITION
    { bufferView := some 1, componentType := 5126, count := nVerts, type := .vec3,
      min := some dmin, max := some dmax },                                         -- 1: POSITION delta
    { bufferView := some 2, componentType := 5125, count := nTris * 3, type := .scalar },
                                                                                    -- 2: INDICES
    { bufferView := some 3, componentType := 5126, count := 3, type := .scalar,
      min := some #[0.0], max := some #[2.0] },                                     -- 3: TIME
    { bufferView := some 4, componentType := 5126, count := 3, type := .scalar,
      min := some #[0.0], max := some #[1.0] }                                      -- 4: WEIGHT
  ]
  let primitive : Primitive := {
    attributes := #[("POSITION", 0)],
    indices := some 2,
    material := some 0,
    targets := #[#[("POSITION", 1)]]
  }
  let mesh : Mesh := {
    primitives := #[primitive],
    name := some "triangleWithSample",
    weights := #[0.0]
  }
  let meshNode : Node := {
    name := some "triangleWithSample.node",
    mesh := some 0
  }
  let anim : Animation := {
    name := some "Profile-Curves rest → rotated → rest (ping-pong)",
    samplers := #[{ input := 3, output := 4, interpolation := .linear }],
    channels := #[{ sampler := 0, target := { node := some 0, path := .weights } }]
  }

  let doc : Document := {
    asset := { generator := some "Curvenet / profile_curves_demo" },
    buffers := #[{ byteLength := bin.size }],
    bufferViews := bufferViews,
    accessors := accessors,
    meshes := #[mesh],
    materials := #[{ name := some "default" }],
    nodes := #[meshNode],
    scenes := #[{ nodes := #[0], name := some "default" }],
    scene := some 0,
    animations := #[anim]
  }
  let glb := LeanGltf.GLB.emit doc bin
  IO.FS.writeBinFile outPath glb
  IO.println s!"wrote {outPath} ({glb.size} bytes; {nVerts} verts, 1 morph target, 2 s ping-pong weight animation)"
  return 0
