/-!
# `Curvenet.SlangValidate.Axpy` — slangc-emitted axpy bridged to Lean

The slangc -target cpp emission of `Curvenet.SlangCodegen.Axpy.shader`
is compiled (via lakefile.lean's `extern_lib axpy_native`) into a
static + shared library `axpy_native`. This module declares the C
entry point as `@[extern]` and exposes a pure-Lean reference for
side-by-side comparison.

End-to-end equivalence is demonstrated by `lake exe axpy_validate`,
which calls both versions on a fixture and asserts equality. Once
Lean's `precompileModules`/dynamic-lookup gap is closed, the same
proof can lift to `native_decide` directly.
-/

namespace Curvenet.SlangValidate.Axpy

/-- Pure-Lean reference: `y[i] := alpha * x[i] + y[i]` for `i < y.size`. -/
def axpySpec (alpha : Float) (x y : FloatArray) : FloatArray := Id.run do
  let n := y.size
  let mut acc : Array Float := y.data
  for i in [:n] do
    if h : i < x.size then
      acc := acc.set! i (alpha * x.get i h + y.get! i)
  return ⟨acc⟩

/-- Slangc-emitted kernel — calls `lean_c_axpy_kernel` (in
    `libaxpy_native`) when the executable is linked against the
    extern_lib; falls back to the pure-Lean spec under the
    interpreter so unit tests in editor mode still type-check. -/
@[extern "lean_c_axpy_kernel"]
def axpyKernel (_n : @& UInt32) (alpha : Float)
               (x y : @& FloatArray) : FloatArray :=
  axpySpec alpha x y

end Curvenet.SlangValidate.Axpy
