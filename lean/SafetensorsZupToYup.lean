import LeanGltf
import LeanSafetensors

/-!
# `safetensors_zup_to_yup` — kernel: bake a Z-up → Y-up rotation into a Mire-shape `.safetensors`

```
lake exe safetensors_zup_to_yup <in.zup.safetensors> <out.yup.safetensors>
```

Why a kernel pass: the previous pipeline put the rotation on a single
root Node and relied on glTF's joint-chain math to push it through. That
works at bind in well-behaved importers but the *buffer data is still in
the wrong frame*. Tools that read POSITION accessors expecting world-y-up
(or that ignore / hoist the root transform) get garbage. Bake the
rotation into every tensor that rides on the axis convention and the .glb
the decoder writes is genuinely Y-up.

Operation, with R = the −90° rotation around X
(Z-up → Y-up; `(x, y, z) → (x, z, -y)`):

  * `mesh.<m>.positions`, `mesh.<m>.normals` (vec3 fp32):
      byte swizzle — X verbatim, Z bytes → Y slot, Y bytes → Z slot with
      sign-bit flip on the high byte. No fp32 decode/encode.

  * `bone.ibm` (mat4 column-major fp32):
      right-multiply each 4×4 by `R⁻¹`. `R⁻¹ = Rᵀ`; in column-major
      this is `[col0, col2, -col1, col3]` per matrix. Each negation is
      a sign-bit flip on the top byte of each fp32. No fp32 decode.

  * `bone.translation` (vec3 fp32, per-bone, parent-local):
      byte-swizzle only the bones whose `bone_parents[i] = -1` (roots).
      Non-root bones are in their parent's local frame; the rotation
      flows through the chain when the root rotates.

  * `bone.rotation` (quat XYZW fp32, per-bone, parent-local):
      for root bones, replace `q` with `q_R · q` where
      `q_R = (-0.7071068, 0, 0, 0.7071068)` (the quaternion for R).
      Required arithmetic — pure byte ops can't do this — so we decode
      / compute / re-encode each fp32 lane.

  * `bone.scale`: unchanged.

  * Metadata `source_axes`: rewritten from `Z-up` to `Y-up`.
-/

open LeanSafetensors

/-! ## fp32 byte codec

Decoder is a port of the helper in `SafetensorsToGltf.lean` (kept private
there). Encoder converts Lean's fp64 `Float` to the IEEE-754 binary32 bit
pattern via `Float.toBits` + manual exponent / mantissa narrowing. We
truncate the bottom 29 bits of the fp64 mantissa rather than rounding to
nearest-even; the resulting ulp drift (≤ 1 ulp at fp32) is negligible for
the rotation we apply (a 0.7071 multiply on already-fp32-precision
quaternion lanes — re-rounding doesn't recover fp32 bit-exactness either
way). NaN / Inf / ±0 / overflow / underflow handled per spec edge cases.
-/

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

private def fp32Encode (f : Float) : UInt32 :=
  let bits64 : UInt64 := f.toBits
  let sign64 : UInt64 := bits64 >>> 63
  let exp64  : UInt64 := (bits64 >>> 52) &&& 0x7FF
  let mant64 : UInt64 := bits64 &&& 0xFFFFFFFFFFFFF
  let sign32 : UInt32 := sign64.toUInt32 <<< 31
  if exp64 == 0x7FF then
    if mant64 ≠ 0 then sign32 ||| 0x7FC00000  -- canonical quiet NaN
    else sign32 ||| 0x7F800000                -- ±Infinity
  else if exp64 == 0 then
    sign32                                    -- ±0 (fp64 subnormals → fp32 zero)
  else
    let unbiased : Int := (exp64.toNat : Int) - 1023
    let biased   : Int := unbiased + 127
    if biased ≤ 0 then sign32                 -- underflow → ±0
    else if biased ≥ 0xFF then sign32 ||| 0x7F800000  -- overflow → ±Inf
    else
      let exp32  : UInt32 := biased.toNat.toUInt32 <<< 23
      let mant32 : UInt32 := (mant64 >>> 29).toUInt32 &&& 0x7FFFFF
      sign32 ||| exp32 ||| mant32

private def pushFp32LE (out : ByteArray) (f : Float) : ByteArray :=
  let bits := fp32Encode f
  out.push  (bits &&& 0xFF).toUInt8
     |>.push ((bits >>> 8)  &&& 0xFF).toUInt8
     |>.push ((bits >>> 16) &&& 0xFF).toUInt8
     |>.push ((bits >>> 24) &&& 0xFF).toUInt8

/-! ## Byte-level vec3 / mat4 swizzles -/

/-- Per vec3 fp32 in `xs`: emit `(X, Z, -Y)` using only byte ops.
    X word verbatim, source Z word verbatim into Y slot, source Y word
    into Z slot with sign-bit XOR on its high byte. -/
private def rotateVec3Bytes (xs : ByteArray) : ByteArray := Id.run do
  let n := xs.size / 12
  let mut out : ByteArray := ByteArray.empty
  for i in [:n] do
    let base := 12 * i
    out := out.push xs[base+0]! |>.push xs[base+1]! |>.push xs[base+2]! |>.push xs[base+3]!
    out := out.push xs[base+8]! |>.push xs[base+9]! |>.push xs[base+10]! |>.push xs[base+11]!
    out := out.push xs[base+4]! |>.push xs[base+5]! |>.push xs[base+6]! |>.push (xs[base+7]! ^^^ 0x80)
  return out

/-- Negate every fp32 in a 16-byte mat4 column (4 floats × 4 bytes).
    Sign-bit flip on the top byte of each. -/
private def negCol (xs : ByteArray) (start : Nat) (out : ByteArray) : ByteArray := Id.run do
  let mut o := out
  for j in [:4] do
    let foff := start + 4 * j
    o := o.push xs[foff]! |>.push xs[foff+1]! |>.push xs[foff+2]! |>.push (xs[foff+3]! ^^^ 0x80)
  return o

private def copyRange (xs : ByteArray) (start len : Nat) (out : ByteArray) : ByteArray := Id.run do
  let mut o := out
  for k in [:len] do
    o := o.push xs[start + k]!
  return o

/-- Right-multiply each column-major mat4 by R⁻¹.
    New columns: [col0, col2, -col1, col3]. -/
private def rotateIbmBytes (xs : ByteArray) : ByteArray := Id.run do
  let n := xs.size / 64   -- 16 fp32 × 4 bytes per matrix
  let mut out : ByteArray := ByteArray.empty
  for i in [:n] do
    let base := 64 * i
    out := copyRange xs base       16 out  -- col 0 verbatim
    out := copyRange xs (base+32)  16 out  -- col 1 = old col 2
    out := negCol    xs (base+16)     out  -- col 2 = -old col 1
    out := copyRange xs (base+48)  16 out  -- col 3 verbatim
  return out

/-- For each per-bone vec3 (12 bytes), if `parents[i] < 0` apply the
    Z-up→Y-up byte swizzle, else copy verbatim. -/
private def rotateRootBoneVec3 (xs : ByteArray) (parents : Array Int) : ByteArray := Id.run do
  let n := parents.size
  let mut out : ByteArray := ByteArray.empty
  for i in [:n] do
    let base := 12 * i
    if parents[i]! < 0 then
      out := out.push xs[base+0]! |>.push xs[base+1]! |>.push xs[base+2]! |>.push xs[base+3]!
      out := out.push xs[base+8]! |>.push xs[base+9]! |>.push xs[base+10]! |>.push xs[base+11]!
      out := out.push xs[base+4]! |>.push xs[base+5]! |>.push xs[base+6]! |>.push (xs[base+7]! ^^^ 0x80)
    else
      out := copyRange xs base 12 out
  return out

/-- For each per-bone quaternion (16 bytes XYZW), if `parents[i] < 0`
    replace with `q_R · q`, else copy verbatim. The product expands as:
      q'.x = c · (qx − qw)
      q'.y = c · (qy + qz)
      q'.z = c · (qz − qy)
      q'.w = c · (qw + qx)
    where `c = √2/2 ≈ 0.7071068` (the only non-zero magnitude in `q_R`). -/
private def rotateRootBoneQuat (xs : ByteArray) (parents : Array Int) : ByteArray := Id.run do
  let n := parents.size
  let c := 0.7071067811865476  -- √2/2 to fp64 precision
  let mut out : ByteArray := ByteArray.empty
  for i in [:n] do
    let base := 16 * i
    if parents[i]! < 0 then
      let qx := readFp32LE xs (base + 0)
      let qy := readFp32LE xs (base + 4)
      let qz := readFp32LE xs (base + 8)
      let qw := readFp32LE xs (base + 12)
      out := pushFp32LE out (c * (qx - qw))
      out := pushFp32LE out (c * (qy + qz))
      out := pushFp32LE out (c * (qz - qy))
      out := pushFp32LE out (c * (qw + qx))
    else
      out := copyRange xs base 16 out
  return out

/-! ## Metadata helper -/

private def parseIntList (src : String) : Except String (Array Int) := do
  let v ← LeanGltf.JSON.parse src
  match v with
  | .arr xs =>
    xs.foldlM (init := (#[] : Array Int)) (fun acc x =>
      match x with
      | .int n => pure (acc.push n)
      | _ => throw "expected int in list")
  | _ => throw "expected JSON array of ints"

/-! ## Main -/

private def isMeshAttr (name : String) (suffix : String) : Bool :=
  name.startsWith "mesh." && name.endsWith ("." ++ suffix)

def main (args : List String) : IO UInt32 := do
  let (inPath, outPath) ← match args with
    | a :: b :: _ => pure (a, b)
    | _ => do IO.eprintln "usage: safetensors_zup_to_yup <in> <out>"; throw (.userError "usage")
  let bytes ← IO.FS.readBinFile inPath
  let st ← match SafeTensors.parse bytes with
    | .ok st => pure st
    | .error e => do IO.eprintln s!"parse: {e}"; return 1

  let parentsStr ← match st.metadataAt "bone_parents" with
    | some s => pure s
    | none => do IO.eprintln "missing metadata: bone_parents"; return 2
  let parents ← match parseIntList parentsStr with
    | .ok a => pure a
    | .error e => do IO.eprintln s!"bone_parents: {e}"; return 2

  let mut entries : Array (String × Dtype × Array Nat × ByteArray) := #[]
  for nt in st.header.tensors do
    let name := nt.1
    let info := nt.2
    let slice := st.body.extract info.offsetLo info.offsetHi
    let newSlice : ByteArray :=
      if info.dtype = .f32 && (isMeshAttr name "positions" || isMeshAttr name "normals") then
        rotateVec3Bytes slice
      else if info.dtype = .f32 && name = "bone.ibm" then
        rotateIbmBytes slice
      else if info.dtype = .f32 && name = "bone.translation" then
        rotateRootBoneVec3 slice parents
      else if info.dtype = .f32 && name = "bone.rotation" then
        rotateRootBoneQuat slice parents
      else
        slice
    entries := entries.push (name, info.dtype, info.shape, newSlice)

  -- Stamp the metadata so the decoder (and any other downstream tool)
  -- can refuse to apply the axis transform a second time.
  let mut newMeta : Array (String × String) := #[]
  let mut sawAxes := false
  for kv in st.header.metadata do
    let k := kv.1
    if k = "source_axes" then
      newMeta := newMeta.push (k, "Y-up")
      sawAxes := true
    else
      newMeta := newMeta.push (k, kv.2)
  if ¬sawAxes then newMeta := newMeta.push ("source_axes", "Y-up")

  let blob ← match build entries newMeta with
    | .ok b => pure b
    | .error e => do IO.eprintln s!"build: {e}"; return 3
  IO.FS.writeBinFile outPath blob
  IO.println s!"kernel zup→yup: {inPath} → {outPath} ({bytes.size} → {blob.size} bytes; {entries.size} tensors)"
  return 0
