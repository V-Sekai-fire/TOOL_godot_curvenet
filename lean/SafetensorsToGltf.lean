import LeanGltf
import LeanSafetensors

/-!
# `safetensors_to_gltf` — assemble a fully-skinned `.glb` from a Mire-style `.safetensors`

```
lake exe safetensors_to_gltf <in.safetensors> <out.glb>
```

Expected input layout (produced by `misc/blend_to_safetensors.py`):

  Tensors, per mesh:
    `mesh.<name>.positions`   F32 [N,3]   per-loop expanded
    `mesh.<name>.normals`     F32 [N,3]
    `mesh.<name>.uvs`         F32 [N,2]   V-flipped to glTF convention
    `mesh.<name>.joints`      U16 [N,4]   global bone palette indices
    `mesh.<name>.weights`     F32 [N,4]   normalised
    `mesh.<name>.indices`     U32 [M,3]   trivial 0..N-1
  Skeleton:
    `bone.ibm`                F32 [B,4,4] column-major mat4 inverse-bind
    `bone.translation`        F32 [B,3]   parent-space rest translation
    `bone.rotation`           F32 [B,4]   parent-space rest quaternion (XYZW)
    `bone.scale`              F32 [B,3]   parent-space rest scale
  Metadata:
    `bone_names`, `bone_parents`, `mesh_names`,
    `mesh_n_verts`, `mesh_n_tris`, `armature_name`, `format`,
    `source_axes` ("Z-up").

We re-use the safetensors body bytes as the glTF BIN chunk verbatim —
the per-tensor `data_offsets` are already aligned to 4-byte boundaries
(every dtype size × inner-shape product is a multiple of 4 here).

Z-up → Y-up is encoded as a root-Node rotation `(-0.7071068, 0, 0, 0.7071068)`
(−90° around X), so vertex / IBM / bone-rest data stays in its native
Z-up frame end-to-end.
-/

open LeanGltf
open LeanGltf.Extensions.EXTStructuralMetadata
open LeanSafetensors

/-! ## fp32 byte-decoding helpers — needed for POSITION min/max + bone TRS -/

private def fp32BitsToFloat (bits : UInt32) : Float :=
  let signBit  : UInt32 := bits >>> 31
  let expBits  : UInt32 := (bits >>> 23) &&& 0xFF
  let mantBits : UInt32 := bits &&& 0x7FFFFF
  let sign : Float := if signBit = 0 then 1.0 else -1.0
  if expBits == 0xFF then 0.0
  else if expBits == 0 then
    sign * mantBits.toNat.toFloat * Float.pow 2.0 (-149.0)
  else
    let exp : Float := expBits.toNat.toFloat - 127.0
    let mant : Float := 1.0 + mantBits.toNat.toFloat * Float.pow 2.0 (-23.0)
    sign * mant * Float.pow 2.0 exp

private def readFp32LE (xs : ByteArray) (off : Nat) : Float :=
  let b0 := xs[off]!.toUInt32
  let b1 := xs[off+1]!.toUInt32
  let b2 := xs[off+2]!.toUInt32
  let b3 := xs[off+3]!.toUInt32
  fp32BitsToFloat (b0 ||| (b1 <<< 8) ||| (b2 <<< 16) ||| (b3 <<< 24))

/-- Componentwise min/max of an `[N, 3]` fp32 buffer. -/
private def vec3MinMax (xs : ByteArray) : Array Float × Array Float := Id.run do
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

/-! ## JSON-list parsing — the metadata fields are JSON-stringified arrays -/

private def parseStringList (src : String) : Except String (Array String) := do
  let v ← LeanGltf.JSON.parse src
  match v with
  | .arr xs =>
    xs.foldlM (init := (#[] : Array String)) (fun acc x =>
      match x with
      | .str s => pure (acc.push s)
      | _ => throw "expected string in list")
  | _ => throw "expected JSON array of strings"

private def parseIntList (src : String) : Except String (Array Int) := do
  let v ← LeanGltf.JSON.parse src
  match v with
  | .arr xs =>
    xs.foldlM (init := (#[] : Array Int)) (fun acc x =>
      match x with
      | .int n => pure (acc.push n)
      | _ => throw "expected int in list")
  | _ => throw "expected JSON array of ints"

/-! ## Curvenet schema (DeGoes22 §3 vocabulary as `EXT_structural_metadata`) -/

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
      ("Curvenet", { properties := #[
        ("name",          str), ("segment_count", nat), ("sample_count",  nat) ] }),
      ("Segment", { properties := #[
        ("p0", f3), ("p1", f3), ("p2", f3), ("p3", f3),
        ("knot_kind_a", nat), ("knot_kind_b", nat) ] }),
      ("Knot", { properties := #[ ("position", f3), ("kind", nat) ] }),
      ("Sample", { properties := #[
        ("position",  f3), ("frame3x3",  f9),
        ("width_lhs", f3), ("width_rhs", f3) ] }),
      ("DDMWeights", { properties := #[
        ("handle_indices", nat), ("weights", f4),
        ("row_offset", nat),     ("row_length", nat) ] })
    ]
  }

/-! ## safetensors → glTF entrypoint -/

private def usage : IO UInt32 := do
  IO.eprintln "usage: safetensors_to_gltf <in.safetensors> <out.glb>"
  return 1

private def safeName (s : String) : String :=
  s.foldl (init := "") (fun acc c =>
    match c with
    | '.' | ' ' | '/' | '[' | ']' => acc.push '_'
    | c => acc.push c)

def main (args : List String) : IO UInt32 := do
  let (inPath, outPath) ← match args with
    | a :: b :: _ => pure (a, b)
    | _ => do let _ ← usage; throw (.userError "usage")

  let bytes ← IO.FS.readBinFile inPath
  let st ← match SafeTensors.parse bytes with
    | .ok st => pure st
    | .error e => do IO.eprintln s!"parse: {e}"; return 1

  -- ## Metadata (bone hierarchy + mesh names)
  let getMeta (k : String) : IO String := do
    match st.metadataAt k with
    | some v => pure v
    | none   => throw (.userError s!"missing metadata key: {k}")

  let boneNamesStr   ← getMeta "bone_names"
  let boneParentsStr ← getMeta "bone_parents"
  let meshNamesStr   ← getMeta "mesh_names"

  let boneNames ← match parseStringList boneNamesStr with
    | .ok a => pure a | .error e => throw (.userError s!"bone_names: {e}")
  let boneParents ← match parseIntList boneParentsStr with
    | .ok a => pure a | .error e => throw (.userError s!"bone_parents: {e}")
  let meshNames ← match parseStringList meshNamesStr with
    | .ok a => pure a | .error e => throw (.userError s!"mesh_names: {e}")

  let nBones := boneNames.size
  let nMeshes := meshNames.size
  if boneParents.size ≠ nBones then
    throw (.userError s!"bone_parents size {boneParents.size} ≠ bone_names size {nBones}")

  -- Refuse Z-up input. The decoder writes a true Y-up `.glb`; its bone
  -- TRS / IBM / vertex data must already be Y-up. Run the upstream
  -- `safetensors_zup_to_yup` kernel first.
  match st.metadataAt "source_axes" with
  | some "Y-up" => pure ()
  | some other =>
    throw (.userError s!"safetensors_to_gltf: refusing source_axes=\"{other}\" input — run `safetensors_zup_to_yup` first")
  | none =>
    -- No metadata at all is acceptable for hand-authored test inputs.
    pure ()

  -- ## State for accessors / bufferViews. We use the safetensors body as
  -- the BIN chunk verbatim; per-tensor offsets/lengths feed bufferViews.
  let mut bufferViews : Array BufferView := #[]
  let mut accessors   : Array Accessor   := #[]

  -- Add a bufferView+accessor pair for tensor `name`. componentType is a
  -- glTF WebGL constant; accType is VEC2/VEC3/VEC4/MAT4/SCALAR. Returns
  -- the new accessor index.
  let addAcc : (name : String) → (compType : Nat) → (accType : AccessorType)
              → (target : Option Nat) → (computeMinMax : Bool)
              → IO (Nat × ByteArray) :=
    fun _ _ _ _ _ => pure (0, ByteArray.empty)  -- placeholder for closure type
  -- (We do the work inline below — Lean can't easily capture mutable
  -- arrays across helper closures, so the loops are explicit.)
  let _ := addAcc

  -- ## Per-mesh accessor bookkeeping: [pos, norm, uv, joints, weights, indices]
  let mut meshes : Array Mesh := #[]
  for i in [:nMeshes] do
    let name := meshNames[i]!
    let safe := safeName name
    let getT (suffix : String) : IO (TensorInfo × ByteArray) := do
      match st.get s!"mesh.{safe}.{suffix}" with
      | some pair => pure pair
      | none => throw (.userError s!"missing tensor mesh.{safe}.{suffix}")

    let (posInfo, posSlice) ← getT "positions"
    let (normInfo, _)       ← getT "normals"
    let (uvInfo, _)         ← getT "uvs"
    let (jntInfo, _)        ← getT "joints"
    let (wgtInfo, _)        ← getT "weights"
    let (idxInfo, _)        ← getT "indices"

    let nVerts := posInfo.numElements / 3
    let nTris  := idxInfo.numElements / 3

    let mkBV (info : TensorInfo) (target : Option Nat) : BufferView :=
      { buffer := 0, byteOffset := info.offsetLo,
        byteLength := info.actualByteSize, target := target }
    let posBV  := bufferViews.size
    bufferViews := bufferViews.push (mkBV posInfo (some BufferView.arrayBuffer))
    let normBV := bufferViews.size
    bufferViews := bufferViews.push (mkBV normInfo (some BufferView.arrayBuffer))
    let uvBV   := bufferViews.size
    bufferViews := bufferViews.push (mkBV uvInfo (some BufferView.arrayBuffer))
    let jntBV  := bufferViews.size
    bufferViews := bufferViews.push (mkBV jntInfo (some BufferView.arrayBuffer))
    let wgtBV  := bufferViews.size
    bufferViews := bufferViews.push (mkBV wgtInfo (some BufferView.arrayBuffer))
    let idxBV  := bufferViews.size
    bufferViews := bufferViews.push (mkBV idxInfo (some BufferView.elementArrayBuffer))

    let (pmin, pmax) := vec3MinMax posSlice
    let posAcc := accessors.size
    accessors := accessors.push {
      bufferView := some posBV, componentType := 5126, count := nVerts,
      type := .vec3, min := some pmin, max := some pmax
    }
    let normAcc := accessors.size
    accessors := accessors.push {
      bufferView := some normBV, componentType := 5126, count := nVerts, type := .vec3
    }
    let uvAcc := accessors.size
    accessors := accessors.push {
      bufferView := some uvBV, componentType := 5126, count := nVerts, type := .vec2
    }
    let jntAcc := accessors.size
    accessors := accessors.push {
      -- 5123 = UNSIGNED_SHORT
      bufferView := some jntBV, componentType := 5123, count := nVerts, type := .vec4
    }
    let wgtAcc := accessors.size
    accessors := accessors.push {
      bufferView := some wgtBV, componentType := 5126, count := nVerts, type := .vec4
    }
    let idxAcc := accessors.size
    accessors := accessors.push {
      -- 5125 = UNSIGNED_INT
      bufferView := some idxBV, componentType := 5125, count := nTris * 3, type := .scalar
    }

    let prim : Primitive := {
      attributes := #[
        ("POSITION",  posAcc),
        ("NORMAL",    normAcc),
        ("TEXCOORD_0", uvAcc),
        ("JOINTS_0",  jntAcc),
        ("WEIGHTS_0", wgtAcc)
      ],
      indices := some idxAcc,
      material := some 0
    }
    meshes := meshes.push { primitives := #[prim], name := some name }

  -- ## Skin's IBM accessor
  let (ibmInfo, _) ← match st.get "bone.ibm" with
    | some p => pure p
    | none => throw (.userError "missing tensor bone.ibm")
  let ibmBV := bufferViews.size
  bufferViews := bufferViews.push {
    buffer := 0, byteOffset := ibmInfo.offsetLo, byteLength := ibmInfo.actualByteSize,
    target := none   -- IBM is not a vertex buffer; no target hint
  }
  let ibmAcc := accessors.size
  accessors := accessors.push {
    bufferView := some ibmBV, componentType := 5126, count := nBones, type := .mat4
  }

  -- ## Bone Nodes (TRS from bone.translation/rotation/scale)
  let (_btInfo, btBytes) ← match st.get "bone.translation" with
    | some p => pure p | none => throw (.userError "missing bone.translation")
  let (_brInfo, brBytes) ← match st.get "bone.rotation" with
    | some p => pure p | none => throw (.userError "missing bone.rotation")
  let (_bsInfo, bsBytes) ← match st.get "bone.scale" with
    | some p => pure p | none => throw (.userError "missing bone.scale")

  -- children-of map: bone i's children = [j | parents[j] == i]
  let mut childrenOfBone : Array (Array Nat) := Array.replicate nBones #[]
  for j in [:nBones] do
    let p := boneParents[j]!
    if p >= 0 && p.toNat < nBones then
      let pi := p.toNat
      childrenOfBone := childrenOfBone.set! pi (childrenOfBone[pi]!.push j)

  -- Bone i lives at node index `i` directly. The buffer data is in
  -- glTF's required Y-up frame (the `safetensors_zup_to_yup` kernel
  -- baked the rotation into POSITION, NORMAL, root-bone TRS, and IBM
  -- tensors), so there's no axis-conversion node to thread bones
  -- under any longer.
  let mut boneNodes : Array Node := #[]
  for i in [:nBones] do
    let tx := readFp32LE btBytes (12*i + 0)
    let ty := readFp32LE btBytes (12*i + 4)
    let tz := readFp32LE btBytes (12*i + 8)
    let rx := readFp32LE brBytes (16*i + 0)
    let ry := readFp32LE brBytes (16*i + 4)
    let rz := readFp32LE brBytes (16*i + 8)
    let rw := readFp32LE brBytes (16*i + 12)
    let sx := readFp32LE bsBytes (12*i + 0)
    let sy := readFp32LE bsBytes (12*i + 4)
    let sz := readFp32LE bsBytes (12*i + 8)
    boneNodes := boneNodes.push {
      name := some boneNames[i]!,
      translation := some #[tx, ty, tz],
      rotation := some #[rx, ry, rz, rw],
      scale := some #[sx, sy, sz],
      children := childrenOfBone[i]!
    }

  -- ## Skin: joint palette = all bone nodes (bone i is at node index i)
  let skinJoints : Array Nat := Array.range nBones
  let firstRootBone : Option Nat :=
    (Array.range nBones).findSome? (fun i =>
      if boneParents[i]! < 0 then some i else none)
  let skin : Skin := {
    joints := skinJoints,
    inverseBindMatrices := some ibmAcc,
    skeleton := firstRootBone,
    name := some "Armature"
  }

  -- ## Mesh nodes — each carries one mesh + a skin reference. Indexed
  -- starting at `nBones` so the bone Nodes occupy [0, nBones).
  let meshNodeOffset : Nat := nBones
  let mut meshNodes : Array Node := #[]
  for i in [:nMeshes] do
    meshNodes := meshNodes.push {
      name := some s!"{meshNames[i]!}.node",
      mesh := some i,
      skin := some 0
    }

  -- ## Scene roots — skeleton root bone(s) + every mesh node, all at
  -- the top level. No axis-conversion wrapper Node.
  let skeletonRoots : Array Nat :=
    Array.range nBones |>.foldl (init := #[]) (fun acc i =>
      if boneParents[i]! < 0 then acc.push i else acc)
  let meshNodeIndices : Array Nat :=
    Array.range nMeshes |>.map (· + meshNodeOffset)
  let sceneRoots : Array Nat := skeletonRoots ++ meshNodeIndices

  let allNodes : Array Node := boneNodes ++ meshNodes

  -- ## Document
  let doc : Document := {
    asset := { generator := some "Curvenet / safetensors_to_gltf" },
    buffers := #[{ byteLength := st.body.size }],
    bufferViews := bufferViews,
    accessors := accessors,
    meshes := meshes,
    skins := #[skin],
    materials := #[{ name := some "default" }],
    nodes := allNodes,
    scenes := #[{ nodes := sceneRoots, name := some "default" }],
    scene := some 0,
    extensions := #[
      (Top.extensionName, Top.toJson { schema := some curvenetSchema })
    ],
    extensionsUsed := #[Top.extensionName]
  }
  let glb := LeanGltf.GLB.emit doc st.body
  IO.FS.writeBinFile outPath glb
  IO.println s!"wrote {outPath} ({glb.size} bytes; {nMeshes} meshes, {nBones} bones)"
  return 0
