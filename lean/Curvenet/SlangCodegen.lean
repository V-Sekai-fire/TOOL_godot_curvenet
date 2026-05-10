import Curvenet.SlangCodegen.DirectDeltaMush
import Curvenet.SlangCodegen.Axpy
import Curvenet.SlangCodegen.AxpyMulti
import Curvenet.SlangCodegen.Saxpby
import Curvenet.SlangCodegen.SaxpbyMulti
import Curvenet.SlangCodegen.Jacobi
import Curvenet.SlangCodegen.JacobiMulti
import Curvenet.SlangCodegen.Spmv
import Curvenet.SlangCodegen.SpmvMulti

/-!
# `Curvenet.SlangCodegen` — Slang shader codegen umbrella

Each submodule produces a `LeanSlang.SlangShaderModule` for one of
the GPU kernels Curvenet's deformer needs. Pinned `native_decide`
fixtures assert the emission text against a hand-checked reference
per kernel.

Layout convention: every kernel module exports

- `shader   : SlangShaderModule`
- `expected : String`
- two `example` lemmas: `emit shader = expected` and
  `shader.entryPointName = "main"`.
-/
