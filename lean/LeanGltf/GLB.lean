import LeanGltf.Document

/-!
# GLB binary container

Layout (glTF 2.0 spec, section 4.4.1):

```
+0x00  magic        u32  little-endian = 0x46546C67  ('glTF')
+0x04  version      u32  little-endian = 2
+0x08  totalLength  u32  little-endian — length of the whole GLB

  --- chunk 0: JSON ---
+0x00  chunkLength  u32  little-endian — padded to 4
+0x04  chunkType    u32  little-endian = 0x4E4F534A   ('JSON')
+0x08  chunkData    bytes (UTF-8 JSON, padded with 0x20)

  --- chunk 1: BIN (optional) ---
+0x00  chunkLength  u32  little-endian — padded to 4
+0x04  chunkType    u32  little-endian = 0x004E4942   ('BIN\0')
+0x08  chunkData    bytes (raw, padded with 0x00)
```

Multibyte values are little-endian here — *opposite* to the PSHome
formats this project parses elsewhere. Don't paste-from-AC11.
-/

namespace LeanGltf.GLB

def magicGLB : UInt32 := 0x46546C67   -- 'glTF'
def chunkJSON : UInt32 := 0x4E4F534A  -- 'JSON'
def chunkBIN  : UInt32 := 0x004E4942  -- 'BIN\0'

/-! ## Little-endian byte writers -/

def pushU32LE (out : ByteArray) (n : UInt32) : ByteArray :=
  out.push  (n &&& 0xFF).toUInt8
     |>.push ((n >>> 8)  &&& 0xFF).toUInt8
     |>.push ((n >>> 16) &&& 0xFF).toUInt8
     |>.push ((n >>> 24) &&& 0xFF).toUInt8

/-! ## Padding helpers -/

/-- Pad a byte array up to the next 4-byte boundary using `pad`.
glTF spec: JSON chunk pads with `0x20` (space), BIN with `0x00`. -/
def padTo4 (data : ByteArray) (pad : UInt8) : ByteArray :=
  let r := data.size % 4
  if r = 0 then data
  else
    let need := 4 - r
    Id.run do
      let mut out := data
      for _ in [:need] do
        out := out.push pad
      pure out

/-! ## Chunk and full-GLB serialiser -/

/-- Build one chunk: 4-byte little-endian length + 4-byte little-endian
type tag + padded data. -/
def emitChunk (chunkType : UInt32) (data : ByteArray) (pad : UInt8) : ByteArray :=
  let padded := padTo4 data pad
  let header := pushU32LE (pushU32LE ByteArray.empty padded.size.toUInt32) chunkType
  header ++ padded

/-- Serialise a glTF document + optional BIN payload into a complete
GLB byte stream. The caller is responsible for ensuring that the BIN
payload's contents match the byteOffsets/byteLengths declared in
`doc.bufferViews` (and that `doc.buffers[0].byteLength = bin.size`). -/
def emit (doc : LeanGltf.Document) (bin : ByteArray) : ByteArray :=
  let json := doc.toJsonString
  let jsonBytes : ByteArray := json.toUTF8
  let jsonChunk := emitChunk chunkJSON jsonBytes 0x20    -- space pad
  let binChunk  := if bin.size = 0 then ByteArray.empty
                   else emitChunk chunkBIN bin 0x00      -- zero pad
  let totalLength : Nat := 12 + jsonChunk.size + binChunk.size
  let header :=
    pushU32LE (pushU32LE (pushU32LE ByteArray.empty magicGLB) 2) totalLength.toUInt32
  header ++ jsonChunk ++ binChunk

/-! ## Reader -/

private def readU32LE (bs : ByteArray) (off : Nat) : Option UInt32 :=
  if off + 4 > bs.size then none
  else
    let b0 := bs[off]!.toUInt32
    let b1 := bs[off+1]!.toUInt32
    let b2 := bs[off+2]!.toUInt32
    let b3 := bs[off+3]!.toUInt32
    some (b0 ||| (b1 <<< 8) ||| (b2 <<< 16) ||| (b3 <<< 24))

private def slice (bs : ByteArray) (off len : Nat) : ByteArray :=
  bs.extract off (off + len)

/-- Parse a GLB byte stream into (JSON string, BIN chunk).
    BIN chunk is empty when the file has no second chunk. Errors on
    bad magic, version, length, or chunk types. -/
def parse (bs : ByteArray) : Except String (String × ByteArray) := do
  if bs.size < 12 then throw "glb: header < 12 bytes"
  let magic ← match readU32LE bs 0 with
    | some n => pure n
    | none   => throw "glb: short read at magic"
  if magic ≠ magicGLB then throw "glb: bad magic"
  let version ← match readU32LE bs 4 with
    | some n => pure n
    | none   => throw "glb: short read at version"
  if version ≠ 2 then throw s!"glb: unsupported version {version}"
  let totalLen ← match readU32LE bs 8 with
    | some n => pure n.toNat
    | none   => throw "glb: short read at totalLength"
  if totalLen > bs.size then throw "glb: declared length exceeds bytes"

  -- Chunk 0: must be JSON.
  let chunk0Len ← match readU32LE bs 12 with
    | some n => pure n.toNat
    | none   => throw "glb: short read at chunk-0 length"
  let chunk0Type ← match readU32LE bs 16 with
    | some n => pure n
    | none   => throw "glb: short read at chunk-0 type"
  if chunk0Type ≠ chunkJSON then throw "glb: chunk 0 is not JSON"
  let chunk0Data := slice bs 20 chunk0Len
  -- The writer appends 0x20 (space) padding to align to 4 bytes; that's
  -- valid JSON whitespace, so we hand the unstripped string straight to
  -- the parser without `trim` (which is being deprecated to a Slice
  -- return type in newer Lean).
  let json := (String.fromUTF8? chunk0Data).getD ""

  -- Chunk 1 (optional): BIN.
  let after0 := 20 + chunk0Len
  if after0 ≥ totalLen then
    return (json, ByteArray.empty)
  if after0 + 8 > bs.size then throw "glb: short read at chunk-1 header"
  let chunk1Len ← match readU32LE bs after0 with
    | some n => pure n.toNat
    | none   => throw "glb: short read at chunk-1 length"
  let chunk1Type ← match readU32LE bs (after0 + 4) with
    | some n => pure n
    | none   => throw "glb: short read at chunk-1 type"
  if chunk1Type ≠ chunkBIN then throw "glb: chunk 1 is not BIN"
  let bin := slice bs (after0 + 8) chunk1Len
  return (json, bin)

end LeanGltf.GLB
