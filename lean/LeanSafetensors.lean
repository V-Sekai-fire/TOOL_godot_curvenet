import LeanSafetensors.Header
import LeanSafetensors.Reader
import LeanSafetensors.Writer

/-!
# `LeanSafetensors` — minimal Lean implementation of the safetensors binary format

```
import LeanSafetensors
open LeanSafetensors

-- Read.
let bytes ← IO.FS.readBinFile "model.safetensors"
let .ok st := SafeTensors.parse bytes | throw …
let some (info, slice) := st.get "weights"

-- Write.
let .ok blob := build #[("weights", .f32, #[4, 4], myFp32Bytes)]
IO.FS.writeBinFile "out.safetensors" blob
```

Spec: <https://github.com/huggingface/safetensors>.
-/
