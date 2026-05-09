/-
MIT License — Copyright (c) 2026 K. S. Ernest (iFire) Lee.

Cubic Bezier specification + concrete property checks, mirroring src/curvenet/bezier.h.

Scope: Lean core (no Mathlib). Float lacks `DecidableEq`, so generic theorems
over `∀ p₀…p₃, t` cannot be closed by `ring`. We instead:

1. Encode the algorithm as Lean `def`s — these ARE the spec; the C++ in
   src/curvenet/bezier.h must agree by inspection.
2. Prove instance theorems via `native_decide` on Boolean equalities over
   concrete control points. These witness that the C++ tests' empirical
   observations hold bit-for-bit on at least the chosen inputs.
3. Mark the lift to ℝ (and full-generality theorems) as Mathlib-future-work.
-/

import Curvenet.Vec3

namespace Curvenet

/-- Bernstein basis polynomials of degree 3 as a tuple. -/
def bezierBasis (t : Float) : Float × Float × Float × Float :=
  let s := 1.0 - t
  (s * s * s, 3.0 * s * s * t, 3.0 * s * t * t, t * t * t)

/-- Cubic Bezier evaluation. Matches `evaluate_cubic_bezier` in C++. -/
def bezier (p0 p1 p2 p3 : Vec3) (t : Float) : Vec3 :=
  let (b0, b1, b2, b3) := bezierBasis t
  b0 * p0 + b1 * p1 + b2 * p2 + b3 * p3

/-- Cubic Bezier derivative. Matches `evaluate_cubic_bezier_derivative` in C++. -/
def bezierDeriv (p0 p1 p2 p3 : Vec3) (t : Float) : Vec3 :=
  let s := 1.0 - t
  (3.0 * s * s) * (p1 - p0) + (6.0 * s * t) * (p2 - p1) + (3.0 * t * t) * (p3 - p2)

/- ============================================================ -/
/- Concrete instance checks via Boolean equality.               -/
/-                                                              -/
/- Each `example` is a propositionally true `Bool = true`,      -/
/- closed by `native_decide` (which executes the Bool expression -/
/- and certifies the result).                                   -/
/- ============================================================ -/

/-- bezier(0) = p₀ on a concrete test point. -/
example :
    (bezier ⟨1.0, 2.0, 3.0⟩ ⟨4.0, 5.0, 6.0⟩ ⟨7.0, 8.0, 9.0⟩ ⟨10.0, 11.0, 12.0⟩ 0.0
        == ⟨1.0, 2.0, 3.0⟩) = true := by
  native_decide

/-- bezier(1) = p₃ on a concrete test point. -/
example :
    (bezier ⟨1.0, 2.0, 3.0⟩ ⟨4.0, 5.0, 6.0⟩ ⟨7.0, 8.0, 9.0⟩ ⟨10.0, 11.0, 12.0⟩ 1.0
        == ⟨10.0, 11.0, 12.0⟩) = true := by
  native_decide

/-- 4 coincident control points → constant curve at the midpoint. -/
example :
    (bezier ⟨1.0, 2.0, 3.0⟩ ⟨1.0, 2.0, 3.0⟩ ⟨1.0, 2.0, 3.0⟩ ⟨1.0, 2.0, 3.0⟩ 0.5
        == ⟨1.0, 2.0, 3.0⟩) = true := by
  native_decide

/-- bezier basis at t=0 is (1, 0, 0, 0). -/
example : (bezierBasis 0.0 == (1.0, 0.0, 0.0, 0.0)) = true := by native_decide

/-- bezier basis at t=1 is (0, 0, 0, 1). -/
example : (bezierBasis 1.0 == (0.0, 0.0, 0.0, 1.0)) = true := by native_decide

/-- Derivative at t=0 equals 3*(p₁ - p₀) on a concrete point. -/
example :
    (bezierDeriv ⟨1.0, 2.0, 3.0⟩ ⟨4.0, 5.0, 6.0⟩ ⟨7.0, 8.0, 9.0⟩ ⟨10.0, 11.0, 12.0⟩ 0.0
        == ⟨9.0, 9.0, 9.0⟩) = true := by
  native_decide

/-- Derivative at t=1 equals 3*(p₃ - p₂) on a concrete point. -/
example :
    (bezierDeriv ⟨1.0, 2.0, 3.0⟩ ⟨4.0, 5.0, 6.0⟩ ⟨7.0, 8.0, 9.0⟩ ⟨10.0, 11.0, 12.0⟩ 1.0
        == ⟨9.0, 9.0, 9.0⟩) = true := by
  native_decide

/- ============================================================ -/
/- Future work: full-generality theorems require ℝ + Mathlib.   -/
/-                                                              -/
/-   theorem bezier_at_zero (p₀ p₁ p₂ p₃ : EuclideanSpace ℝ 3)  -/
/-       : bezier p₀ p₁ p₂ p₃ 0 = p₀ := by                      -/
/-     simp [bezier, bezierBasis]; ring                         -/
/-                                                              -/
/-   theorem bezier_basis_partition (t : ℝ)                     -/
/-       : (bezierBasis t).1 + ... = 1 := by                    -/
/-     simp [bezierBasis]; ring                                 -/
/- ============================================================ -/

end Curvenet
