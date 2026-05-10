import LeanGltf.JSON
import LeanGltf.Types
import LeanGltf.Document
import LeanGltf.GLB
import LeanGltf.Writer

/-!
# `LeanGltf` — umbrella import

Writer-only port of the glTF 2.0 graph from `aria-gltf` (Elixir, MIT,
K. S. Ernest "iFire" Lee). Use this module to pull in the full surface.

```
import LeanGltf
open LeanGltf
let doc : Document := { … }
writeGlb "out.glb" doc bin
```
-/
