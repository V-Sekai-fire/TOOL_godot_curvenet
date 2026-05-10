import LeanSafetensors

/-!
# `safetensors_roundtrip` — read a `.safetensors`, re-emit it

```
lake exe safetensors_roundtrip <in.safetensors> <out.safetensors>
```

The output is byte-equivalent in *content* (same tensor data, same
metadata, same dtype/shape per tensor) but not necessarily byte-equal
(JSON key ordering, header padding may differ vs the reference Python
`safetensors` library). The companion Python verifier in
`tests/safetensors/run_tests.py` checks content equivalence.
-/

open LeanSafetensors

def main (args : List String) : IO UInt32 := do
  match args with
  | inPath :: outPath :: _ => do
    let bytes ← IO.FS.readBinFile inPath
    match SafeTensors.parse bytes with
    | .error e => do IO.eprintln s!"parse error: {e}"; return 1
    | .ok st => do
      -- Re-emit. Tensor order and metadata order are preserved from
      -- the input header, so the rewrite is faithful modulo JSON
      -- quirks (key separators, padding count).
      let entries : Array (String × Dtype × Array Nat × ByteArray) :=
        st.header.tensors.map (fun nt =>
          let name := nt.1
          let info := nt.2
          let slice := st.body.extract info.offsetLo info.offsetHi
          (name, info.dtype, info.shape, slice))
      match build entries st.header.metadata with
      | .error e => do IO.eprintln s!"build error: {e}"; return 2
      | .ok blob => do
        IO.FS.writeBinFile outPath blob
        IO.println s!"roundtrip {inPath} -> {outPath} ({bytes.size} -> {blob.size} bytes; {st.header.tensors.size} tensors, {st.header.metadata.size} metadata)"
        return 0
  | _ => do
    IO.eprintln "usage: safetensors_roundtrip <in> <out>"
    return 1
