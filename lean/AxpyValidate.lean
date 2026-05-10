import Curvenet.SlangValidate.Axpy

open Curvenet.SlangValidate.Axpy

/-! Equivalence runner: spec ≡ slangc-emitted CPU kernel on a fixture.
    Linked against the `axpy_native` extern_lib so `axpyKernel` calls
    the slangc-cpp output rather than the Lean fallback. -/

def floatArrayApproxEq (a b : FloatArray) (eps : Float) : Bool := Id.run do
  if a.size != b.size then return false
  let mut ok := true
  for i in [:a.size] do
    let d := a.get! i - b.get! i
    let abs := if d < 0.0 then -d else d
    if abs > eps then ok := false
  return ok

def runFixture
    (label : String) (alpha : Float)
    (xs ys : Array Float) : IO Bool := do
  let x : FloatArray := ⟨xs⟩
  let y : FloatArray := ⟨ys⟩
  let n := xs.size.toUInt32
  let viaKernel := axpyKernel n alpha x y
  let viaSpec   := axpySpec alpha x y
  let agree := floatArrayApproxEq viaKernel viaSpec 1e-6
  if agree then
    IO.println s!"{label}: kernel ≡ spec  ({n} elems, alpha={alpha})"
    return true
  else
    IO.eprintln s!"{label}: MISMATCH"
    IO.eprintln s!"  spec   = {viaSpec.data}"
    IO.eprintln s!"  kernel = {viaKernel.data}"
    return false

def main : IO UInt32 := do
  let mut fails := 0
  unless (← runFixture "axpy/n=4 alpha=2"  2.0
            #[1.0, 2.0, 3.0, 4.0] #[10.0, 20.0, 30.0, 40.0]) do
    fails := fails + 1
  unless (← runFixture "axpy/n=8 alpha=-1" (-1.0)
            #[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
            #[100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]) do
    fails := fails + 1
  unless (← runFixture "axpy/n=1 alpha=0"  0.0 #[42.0] #[7.0]) do
    fails := fails + 1
  if fails == 0 then
    IO.println "all axpy equivalence checks PASS"
    return 0
  else
    IO.eprintln s!"{fails} fixture(s) FAILED"
    return 1
