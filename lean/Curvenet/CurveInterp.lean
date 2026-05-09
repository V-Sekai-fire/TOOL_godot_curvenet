/-
MIT License — Copyright (c) 2026 K. S. Ernest (iFire) Lee.

Per-side normal/width interpolation along a curve — slice 16 of the
DeGoes22 rewrite.

DeGoes22 §3 ("Normal & width interpolation"):
  Given per-side `(n_i^±, w_i^±)` at the two intersection-adjacent
  endpoints of a curve, fill the values at intermediate segments by
  interpolating along the curve. The paper does not pin down the
  interpolant; we use linear blend (no renormalization), which is
  sufficient for the corner-recovery checks the slice-20 cut-mesh tests
  will rely on. The full algorithm-stack picks slerp later if a
  follow-up needs it; for now linear keeps the proofs decidable.

The intersection-adjacent endpoint values come from `IntersectionFrames`
(slice 13). For curves that are isolated (no intersections — closed
loops or anchor-anchor pairs), the simpler slice-4 isolated-segment
formulation applies and `interpAlongCurve` is not used.
-/

import Curvenet.Vec3

namespace Curvenet
namespace CurveInterp

/-- Per-side state at one segment: a normal vector and a width scalar. -/
structure SideData where
  n : Vec3
  w : Float
deriving Repr, Inhabited

@[inline] def lerpVec3 (a b : Vec3) (t : Float) : Vec3 :=
  ⟨ (1.0 - t) * a.x + t * b.x
  , (1.0 - t) * a.y + t * b.y
  , (1.0 - t) * a.z + t * b.z ⟩

@[inline] def lerpFloat (a b : Float) (t : Float) : Float :=
  (1.0 - t) * a + t * b

@[inline] def lerpSide (a b : SideData) (t : Float) : SideData :=
  { n := lerpVec3 a.n b.n t
  , w := lerpFloat a.w b.w t }

/-- Linear interpolation of per-side `(n^+, n^-, w^+, w^-)` along a curve
   over `n` segments. Index 0 carries `first`, index n−1 carries `last`,
   intermediate indices linearly blend.

   Inputs `first` and `last` are pairs `(plusSide, minusSide)`.
   Output is an array of the same shape, length `n`. -/
def interpAlongCurve (first last : SideData × SideData) (n : Nat) :
    Array (SideData × SideData) := Id.run do
  if n = 0 then return #[]
  if n = 1 then return #[first]
  let mut acc : Array (SideData × SideData) := Array.replicate n first
  let denom := Float.ofNat (n - 1)
  let (firstP, firstM) := first
  let (lastP,  lastM)  := last
  for i in [0:n] do
    let t := Float.ofNat i / denom
    acc := acc.set! i (lerpSide firstP lastP t, lerpSide firstM lastM t)
  return acc

end CurveInterp

/- ============================================================ -/
/- Concrete checks: 2- and 3-segment curves between two known   -/
/- intersection endpoints.                                      -/
/- ============================================================ -/

namespace CurveInterpExamples

open CurveInterp

/-- Sample endpoints. First curve segment: n^+ = +z, n^- = −z, w^+ = 1, w^- = 1.
   Last curve segment: n^+ = +x, n^- = −x, w^+ = 5, w^- = 3. -/
private def first : SideData × SideData :=
  ( ⟨⟨0.0, 0.0, 1.0⟩, 1.0⟩
  , ⟨⟨0.0, 0.0, -1.0⟩, 1.0⟩ )

private def last : SideData × SideData :=
  ( ⟨⟨1.0, 0.0, 0.0⟩, 5.0⟩
  , ⟨⟨-1.0, 0.0, 0.0⟩, 3.0⟩ )

/-- 2-segment curve (just the two endpoints, no intermediate to blend). -/
example :
    let arr := interpAlongCurve first last 2
    let (p0, m0) := arr[0]!
    let (p1, m1) := arr[1]!
    -- index 0 reproduces `first`, index 1 reproduces `last`.
    ((p0.w - 1.0).abs < 1e-12 ∧ (p0.n.z - 1.0).abs < 1e-12 ∧
     (p1.w - 5.0).abs < 1e-12 ∧ (p1.n.x - 1.0).abs < 1e-12 ∧
     (m0.n.z + 1.0).abs < 1e-12 ∧
     (m1.n.x + 1.0).abs < 1e-12) := by
  native_decide

/-- 3-segment curve: midpoint linearly blends the two endpoints. -/
example :
    let arr := interpAlongCurve first last 3
    let (p1, m1) := arr[1]!
    -- p1.n = 0.5·(0,0,1) + 0.5·(1,0,0) = (0.5, 0, 0.5)
    -- p1.w = 0.5·1 + 0.5·5 = 3
    -- m1.n = 0.5·(0,0,−1) + 0.5·(−1,0,0) = (−0.5, 0, −0.5)
    -- m1.w = 0.5·1 + 0.5·3 = 2
    ((p1.n.x - 0.5).abs < 1e-12 ∧ p1.n.y.abs < 1e-12 ∧ (p1.n.z - 0.5).abs < 1e-12 ∧
     (p1.w - 3.0).abs < 1e-12 ∧
     (m1.n.x + 0.5).abs < 1e-12 ∧ (m1.n.z + 0.5).abs < 1e-12 ∧
     (m1.w - 2.0).abs < 1e-12) := by
  native_decide

/-- 5-segment curve: index 2 (centre) lands at t = 0.5; index 1 at t = 0.25. -/
example :
    let arr := interpAlongCurve first last 5
    let (p1, _) := arr[1]!
    let (p2, _) := arr[2]!
    -- p1 at t = 0.25: w = 0.75·1 + 0.25·5 = 2
    -- p2 at t = 0.5:  w = 0.5·1 + 0.5·5 = 3
    ((p1.w - 2.0).abs < 1e-12 ∧ (p2.w - 3.0).abs < 1e-12) := by
  native_decide

/-- Edge case n = 1: only the first endpoint is returned. -/
example :
    let arr := interpAlongCurve first last 1
    arr.size = 1 := by native_decide

/-- Edge case n = 0: empty output. -/
example :
    let arr := interpAlongCurve first last 0
    arr.size = 0 := by native_decide

end CurveInterpExamples

end Curvenet
