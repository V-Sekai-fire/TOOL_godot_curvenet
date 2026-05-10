import LeanGltf.Types

/-!
# Document container

Top-level glTF object that bundles every array referenced by index from
the other graph nodes. Mirrors `aria-gltf`'s `AriaGltf.Document.to_json/1`.
-/

namespace LeanGltf
open LeanGltf.JSON

structure Document where
  asset              : Asset := {}
  scene              : Option Nat := some 0
  scenes             : Array Scene       := #[]
  nodes              : Array Node        := #[]
  meshes             : Array Mesh        := #[]
  skins              : Array Skin        := #[]
  animations         : Array Animation   := #[]
  accessors          : Array Accessor    := #[]
  bufferViews        : Array BufferView  := #[]
  buffers            : Array Buffer      := #[]
  materials          : Array Material    := #[]
  /-- Document-level extensions: array of (name, JSON-object) pairs
      embedded into the top-level `"extensions"` object. -/
  extensions         : Array (String × JSON.Value) := #[]
  /-- Names of extensions referenced anywhere in the document. glTF
      readers use this to short-circuit on extensions they don't know. -/
  extensionsUsed     : Array String := #[]
  /-- Subset of `extensionsUsed` that the document cannot be loaded
      without. Readers MUST refuse the file if any of these are
      unknown. -/
  extensionsRequired : Array String := #[]

namespace Document

/-- Render the document to a JSON string. The order of keys matches what
glTF exporters typically produce (asset first, scene/scenes, nodes, etc.) —
not load-bearing, but makes diffs against reference exporters easier. -/
def toJsonString (d : Document) : String :=
  let stringArr (xs : Array String) : JSON.Value :=
    .arr (xs.map (fun s => .str s))
  let entries : Array (String × Option JSON.Value) := #[
    ("asset",              some (Asset.toJson d.asset)),
    ("scene",              d.scene.map (fun n => .int (Int.ofNat n))),
    ("scenes",             if d.scenes.isEmpty      then none else some (.arr (d.scenes.map      Scene.toJson))),
    ("nodes",              if d.nodes.isEmpty       then none else some (.arr (d.nodes.map       Node.toJson))),
    ("meshes",             if d.meshes.isEmpty      then none else some (.arr (d.meshes.map      Mesh.toJson))),
    ("skins",              if d.skins.isEmpty       then none else some (.arr (d.skins.map       Skin.toJson))),
    ("animations",         if d.animations.isEmpty  then none else some (.arr (d.animations.map  Animation.toJson))),
    ("accessors",          if d.accessors.isEmpty   then none else some (.arr (d.accessors.map   Accessor.toJson))),
    ("bufferViews",        if d.bufferViews.isEmpty then none else some (.arr (d.bufferViews.map BufferView.toJson))),
    ("buffers",            if d.buffers.isEmpty     then none else some (.arr (d.buffers.map     Buffer.toJson))),
    ("materials",          if d.materials.isEmpty   then none else some (.arr (d.materials.map   Material.toJson))),
    ("extensions",         if d.extensions.isEmpty  then none else some (.obj d.extensions)),
    ("extensionsUsed",     if d.extensionsUsed.isEmpty     then none else some (stringArr d.extensionsUsed)),
    ("extensionsRequired", if d.extensionsRequired.isEmpty then none else some (stringArr d.extensionsRequired))
  ]
  JSON.render (JSON.obj? entries)

end Document
end LeanGltf
