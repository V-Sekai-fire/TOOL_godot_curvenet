import LeanGltf.GLB

/-!
# `LeanGltf.Writer`

Top-level IO entry points.
-/

namespace LeanGltf

/-- Write a `(Document, BIN)` pair as a GLB to `path`. -/
def writeGlb (path : System.FilePath) (doc : Document) (bin : ByteArray)
    : IO Unit := do
  let bytes := GLB.emit doc bin
  IO.FS.writeBinFile path bytes

/-- Convenience: render only the JSON-text form of a glTF document
(no BIN). Useful for diffing against reference exporters. -/
def writeGltfJson (path : System.FilePath) (doc : Document) : IO Unit := do
  IO.FS.writeFile path (doc.toJsonString)

end LeanGltf
