import LeanGltf

/-!
# `gltf_inspect` — read a `.glb`, parse it, print a summary

The reader counterpart to `safetensors_to_gltf`. Demonstrates that
LeanGltf can ingest `.glb` files, parse the JSON chunk into our
`Value` AST, and surface key counts (mesh / accessor / extension
classes) without a full glTF round-trip.

Usage:

    cd lean && lake exe gltf_inspect /path/to/asset.glb
-/

open LeanGltf
open LeanGltf.JSON (Value)

/-- Look up a string-keyed field in a JSON object. -/
private def lookup (kvs : Array (String × Value)) (key : String) : Option Value :=
  match kvs.findSome? (fun (k, v) => if k = key then some v else none) with
  | some v => some v
  | none   => none

/-- The element count of an array, or 0 for any non-array. -/
private def arrSize : Value → Nat
  | .arr xs => xs.size
  | _       => 0

/-- Pull a List of String keys from a Value object. -/
private def objKeys : Value → List String
  | .obj kvs => kvs.toList.map (·.1)
  | _        => []

def main (args : List String) : IO UInt32 := do
  match args with
  | path :: _ => do
    let bs ← IO.FS.readBinFile path
    match LeanGltf.GLB.parse bs with
    | .error e => do IO.eprintln s!"parse error: {e}"; return 1
    | .ok (jsonStr, bin) => do
      IO.println s!"glb size:   {bs.size} bytes"
      IO.println s!"json chunk: {jsonStr.length} chars"
      IO.println s!"bin  chunk: {bin.size} bytes"

      match LeanGltf.JSON.parse jsonStr with
      | .error e => do IO.eprintln s!"json error: {e}"; return 2
      | .ok val => do
        match val with
        | .obj entries => do
          let keys := entries.toList.map (·.1)
          IO.println ""
          IO.println s!"top-level keys: {keys}"

          -- Counts for the standard arrays.
          for k in ["scenes", "nodes", "meshes", "materials",
                    "accessors", "bufferViews", "buffers"] do
            if let some v := lookup entries k then
              IO.println s!"  {k}: {arrSize v}"

          -- Extensions.
          if let some (.arr used) := lookup entries "extensionsUsed" then
            let names := used.toList.filterMap (fun
              | .str s => some s
              | _ => none)
            IO.println s!"\nextensionsUsed: {names}"
          if let some (.obj exts) := lookup entries "extensions" then
            for (extName, extVal) in exts do
              IO.println s!"\n=== extension: {extName} ==="
              match extVal with
              | .obj ext =>
                if let some schema := lookup ext "schema" then
                  match schema with
                  | .obj sch =>
                    if let some (.str id) := lookup sch "id" then
                      IO.println s!"  schema id:   {id}"
                    if let some (.str nm) := lookup sch "name" then
                      IO.println s!"  schema name: {nm}"
                    if let some classes := lookup sch "classes" then
                      IO.println s!"  classes:     {objKeys classes}"
                  | _ => pure ()
                IO.println s!"  top fields:  {ext.toList.map (·.1)}"
              | _ =>
                IO.println "  (extension is not an object)"

          -- POSITION accessor min/max read-back.
          match lookup entries "accessors" with
          | some (.arr accs) =>
            if accs.size > 0 then
              if let .obj a0 := accs[0]! then
                if let some (.arr mn) := lookup a0 "min" then
                  if let some (.arr mx) := lookup a0 "max" then
                    let toFloats (xs : Array Value) : List Float :=
                      xs.toList.filterMap (fun
                        | .num f => some f
                        | .int i => some (Float.ofInt i)
                        | _ => none)
                    IO.println ""
                    IO.println s!"accessor[0] min: {toFloats mn}"
                    IO.println s!"accessor[0] max: {toFloats mx}"
          | _ => pure ()
          return 0
        | _ => do
          IO.eprintln "json root is not an object"
          return 3
  | _ => do
    IO.eprintln "usage: gltf_inspect <path.glb>"
    return 1
