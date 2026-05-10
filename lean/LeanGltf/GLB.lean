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

end LeanGltf.GLB
