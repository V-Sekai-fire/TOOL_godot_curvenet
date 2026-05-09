/-
MIT License — Copyright (c) 2026 K. S. Ernest (iFire) Lee.

Kernel-mode projection for V-cycle corrections.

Loop-100 diagnosis: at 81k the multilevel V-cycle stalls at
relative residual ~6e-3 even with always-connected HEM aggregates.
The dynamics (residual drops 5 orders in 20 iters then decays at
~0.99/iter for hundreds of iters) is the signature of constant-
mode drift: the prolonged coarse correction `R^T y_c` is piecewise
constant per cluster and its mean is rarely zero, so each outer
iter shifts the fine solution by `mean(y)` in the constant mode.
With Tikhonov-mollified diagonal the constant mode is near-null,
so this drift takes hundreds of iters to bleed off through the
regularization.

The fix is dirt cheap and load-bearing: project the V-cycle
correction onto the orthogonal complement of the constant mode
before applying it.

  zeroMean v   = v - (sum v / |v|) * 1

Properties verified by native_decide:
  * sum (zeroMean v) ≈ 0
  * idempotent: zeroMean (zeroMean v) ≈ zeroMean v
  * a constant vector projects to zero
  * a vector already orthogonal to 1 is unchanged
-/

namespace Curvenet
namespace KernelProjection

/-- Subtract the mean of `v` from every entry. Result satisfies
   `sum (zeroMean v) = 0` (modulo floating-point error). -/
def zeroMean (v : Array Float) : Array Float :=
  let n := v.size
  if n = 0 then v
  else
    let s := v.foldl (· + ·) 0.0
    let m := s / Float.ofNat n
    v.map (fun x => x - m)

/-- Sum of an array of Floats. -/
def sumF (v : Array Float) : Float := v.foldl (· + ·) 0.0

/-- Tolerance-aware Float equality. -/
def fclose (x y eps : Float) : Bool := (x - y).abs < eps

end KernelProjection

/- ============================================================ -/
/- Native-decide checks on small vectors.                        -/
/- ============================================================ -/

namespace KernelProjectionExamples

open KernelProjection

/-- 4-element vector `[1, 2, 3, 4]` has mean 2.5; zero-mean version
   is `[-1.5, -0.5, 0.5, 1.5]`. -/
example :
    fclose (sumF (zeroMean #[1.0, 2.0, 3.0, 4.0])) 0.0 1e-12 = true := by
  native_decide

example :
    let z := zeroMean #[1.0, 2.0, 3.0, 4.0]
    (fclose z[0]! (-1.5) 1e-12 ∧
     fclose z[1]! (-0.5) 1e-12 ∧
     fclose z[2]!   0.5  1e-12 ∧
     fclose z[3]!   1.5  1e-12) = true := by
  native_decide

/-- A constant vector projects to zero. -/
example :
    let z := zeroMean #[7.0, 7.0, 7.0, 7.0, 7.0]
    (fclose (sumF z) 0.0 1e-12 ∧
     fclose z[0]! 0.0 1e-12 ∧ fclose z[1]! 0.0 1e-12 ∧
     fclose z[2]! 0.0 1e-12 ∧ fclose z[3]! 0.0 1e-12 ∧
     fclose z[4]! 0.0 1e-12) = true := by
  native_decide

/-- A vector already with zero mean is unchanged (within eps). -/
example :
    let v := #[(-1.5 : Float), -0.5, 0.5, 1.5]
    let z := zeroMean v
    (fclose z[0]! v[0]! 1e-12 ∧ fclose z[1]! v[1]! 1e-12 ∧
     fclose z[2]! v[2]! 1e-12 ∧ fclose z[3]! v[3]! 1e-12) = true := by
  native_decide

/-- Idempotent: zeroMean (zeroMean v) = zeroMean v. -/
example :
    let v := #[(2.5 : Float), -3.0, 1.0, 4.5, -7.0]
    let z := zeroMean v
    let zz := zeroMean z
    (fclose zz[0]! z[0]! 1e-12 ∧ fclose zz[1]! z[1]! 1e-12 ∧
     fclose zz[2]! z[2]! 1e-12 ∧ fclose zz[3]! z[3]! 1e-12 ∧
     fclose zz[4]! z[4]! 1e-12) = true := by
  native_decide

/-- Empty input passes through unchanged. -/
example : zeroMean (#[] : Array Float) = #[] := by native_decide

end KernelProjectionExamples

end Curvenet
