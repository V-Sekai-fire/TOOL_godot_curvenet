import LeanSafetensors.Header

/-!
# safetensors writer — emit `(name, dtype, shape, bytes)` tuples
-/

namespace LeanSafetensors

private def pushU64LE (out : ByteArray) (n : Nat) : ByteArray := Id.run do
  let mut o := out
  let mut x : Nat := n
  for _ in [:8] do
    o := o.push (UInt8.ofNat (x % 256))
    x := x / 256
  return o

/-- Compute (header, body, byte_layout) from `(name, dtype, shape, bytes)`
    tuples. Tensor offsets are assigned contiguously in input order.
    Throws if `bytes.size ≠ dtype.byteSize · ∏ shape` for any entry. -/
def layout
    (entries  : Array (String × Dtype × Array Nat × ByteArray))
    (metadata : Array (String × String) := #[])
    : Except String (Header × ByteArray) := do
  let mut offsetCursor : Nat := 0
  let mut tensors : Array (String × TensorInfo) := #[]
  let mut body : ByteArray := ByteArray.empty
  for entry in entries do
    let name  := entry.1
    let dtype := entry.2.1
    let shape := entry.2.2.1
    let data  := entry.2.2.2
    let info : TensorInfo :=
      { dtype := dtype, shape := shape
      , offsetLo := offsetCursor
      , offsetHi := offsetCursor + data.size }
    if info.actualByteSize ≠ info.expectedByteSize then
      throw s!"safetensors: tensor {name} bytes {data.size} ≠ dtype·shape product {info.expectedByteSize}"
    tensors := tensors.push (name, info)
    body := body ++ data
    offsetCursor := offsetCursor + data.size
  pure ({ tensors := tensors, metadata := metadata }, body)

/-- Encode a fully-laid-out `(Header, body)` pair as the on-disk byte
    stream. The JSON header is space-padded so the body starts on an
    8-byte boundary — matches what `safetensors.torch.save_file` does
    and keeps mmap-style consumers aligned. -/
def encode (h : Header) (body : ByteArray) : ByteArray := Id.run do
  let mut hdrStr := h.toJsonString
  let preludeSize := 8 + hdrStr.toUTF8.size
  let pad := (8 - preludeSize % 8) % 8
  for _ in [:pad] do hdrStr := hdrStr.push ' '
  let hdrBytes := hdrStr.toUTF8
  let mut out := pushU64LE ByteArray.empty hdrBytes.size
  out := out ++ hdrBytes
  out := out ++ body
  return out

/-- Convenience one-shot: lay out tensors + emit the byte stream. -/
def build
    (entries  : Array (String × Dtype × Array Nat × ByteArray))
    (metadata : Array (String × String) := #[])
    : Except String ByteArray := do
  let (h, body) ← layout entries metadata
  pure (encode h body)

end LeanSafetensors
