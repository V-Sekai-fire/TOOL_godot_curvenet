import LeanSafetensors.Header
import LeanGltf.JSON

/-!
# safetensors reader — parse a `.safetensors` byte stream
-/

namespace LeanSafetensors

structure SafeTensors where
  header : Header
  /-- Tensor data section (everything after the JSON header). Tensor
      offsets in `header.tensors` are relative to byte 0 of `body`. -/
  body   : ByteArray
  deriving Inhabited

namespace SafeTensors

private def readU64LE (bs : ByteArray) (off : Nat) : Option Nat :=
  if off + 8 > bs.size then none
  else Id.run do
    let mut acc : Nat := 0
    for i in [:8] do
      acc := acc + (bs[off + i]!.toNat <<< (8*i))
    return some acc

/-- Validate that every declared tensor's byte slice is in-bounds and
    matches `dtype.byteSize · numElements`. Catches truncated files
    and shape/dtype/offset disagreement. -/
def validate (st : SafeTensors) : Except String Unit := do
  for nt in st.header.tensors do
    let name := nt.1
    let t    := nt.2
    if t.offsetLo > t.offsetHi then
      throw s!"safetensors: tensor {name} has lo > hi"
    if t.offsetHi > st.body.size then
      throw s!"safetensors: tensor {name} extends beyond body ({t.offsetHi} > {st.body.size})"
    let expected := t.expectedByteSize
    if t.actualByteSize ≠ expected then
      throw s!"safetensors: tensor {name} size {t.actualByteSize} ≠ dtype·shape product {expected}"
  pure ()

/-- Parse a safetensors byte stream. Runs `validate` before returning. -/
def parse (bs : ByteArray) : Except String SafeTensors := do
  if bs.size < 8 then throw "safetensors: file < 8 bytes"
  let hlen ← match readU64LE bs 0 with
    | some n => pure n
    | none   => throw "safetensors: short read at header length"
  if 8 + hlen > bs.size then
    throw s!"safetensors: header length {hlen} exceeds file size {bs.size}"
  let hdrBytes := bs.extract 8 (8 + hlen)
  let hdrStr := (String.fromUTF8? hdrBytes).getD ""
  let json ← LeanGltf.JSON.parse hdrStr
  let header ← Header.fromJson json
  let body := bs.extract (8 + hlen) bs.size
  let st : SafeTensors := { header := header, body := body }
  validate st
  pure st

/-- Look up a tensor by name. Returns `(info, slice)` where `slice` is
    a *new* ByteArray over the tensor's bytes (no aliasing). -/
def get (st : SafeTensors) (name : String) : Option (TensorInfo × ByteArray) :=
  st.header.tensors.findSome? (fun (n, t) =>
    if n = name then
      some (t, st.body.extract t.offsetLo t.offsetHi)
    else none)

/-- Look up a string entry from `__metadata__`. -/
def metadataAt (st : SafeTensors) (key : String) : Option String :=
  st.header.metadata.findSome? (fun (k, v) =>
    if k = key then some v else none)

end SafeTensors
end LeanSafetensors
