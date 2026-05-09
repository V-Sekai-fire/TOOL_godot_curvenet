/-
MIT License — Copyright (c) 2026 K. S. Ernest (iFire) Lee.

N-sided patch dispatcher mirroring src/curvenet/ngon_patch.h:
  N=3 -> degenerate Coons (top edge collapsed to P2)
  N=4 -> standard CoonsPatch
  N>=5 -> reserved for mean-value-coords blending (TODO)
-/

import Curvenet.Vec3
import Curvenet.Bezier
import Curvenet.CoonsPatch

namespace Curvenet

/-- Reverse the orientation of a boundary curve. -/
def BoundaryCurve.reverse (b : BoundaryCurve) : BoundaryCurve :=
  { c0 := b.c3, c1 := b.c2, c2 := b.c1, c3 := b.c0 }

/-- Build a degenerate-Coons triangular patch from 3 CCW boundaries. -/
def triPatch (b0 b1 b2 : BoundaryCurve) : CoonsPatch :=
  let P2 := b1.c3 -- = b2.c0
  { u0 := b0
  , v1 := b1
  , v0 := b2.reverse
  , u1 := { c0 := P2, c1 := P2, c2 := P2, c3 := P2 }
  }

/-- 3-sided patch evaluation. -/
def triEvaluate (b0 b1 b2 : BoundaryCurve) (s t : Float) : Vec3 :=
  (triPatch b0 b1 b2).evaluate s t

/- ============================================================ -/
/- Concrete corner-recovery checks for the triangular path.    -/
/- ============================================================ -/

private def triBoundaries : BoundaryCurve × BoundaryCurve × BoundaryCurve :=
  let P0 : Vec3 := ⟨0.0, 0.0, 0.0⟩
  let P1 : Vec3 := ⟨1.0, 0.0, 0.0⟩
  let P2 : Vec3 := ⟨0.5, 1.0, 0.0⟩
  let lerp : Vec3 → Vec3 → Float → Vec3 := fun a b t => (1.0 - t) * a + t * b
  ( { c0 := P0, c1 := lerp P0 P1 (1.0/3.0), c2 := lerp P0 P1 (2.0/3.0), c3 := P1 }
  , { c0 := P1, c1 := lerp P1 P2 (1.0/3.0), c2 := lerp P1 P2 (2.0/3.0), c3 := P2 }
  , { c0 := P2, c1 := lerp P2 P0 (1.0/3.0), c2 := lerp P2 P0 (2.0/3.0), c3 := P0 }
  )

/-- Corner (0,0) interpolates P0. -/
example :
    let (b0, b1, b2) := triBoundaries
    (triEvaluate b0 b1 b2 0.0 0.0 == ⟨0.0, 0.0, 0.0⟩) = true := by
  native_decide

/-- Corner (1,0) interpolates P1. -/
example :
    let (b0, b1, b2) := triBoundaries
    (triEvaluate b0 b1 b2 1.0 0.0 == ⟨1.0, 0.0, 0.0⟩) = true := by
  native_decide

/-- Top edge (anything, 1) collapses to P2. -/
example :
    let (b0, b1, b2) := triBoundaries
    (triEvaluate b0 b1 b2 0.0 1.0 == ⟨0.5, 1.0, 0.0⟩) = true := by
  native_decide

/-- Top edge with s = 0.5 still equals P2 (degenerate). -/
example :
    let (b0, b1, b2) := triBoundaries
    (triEvaluate b0 b1 b2 0.5 1.0 == ⟨0.5, 1.0, 0.0⟩) = true := by
  native_decide

/-- Top edge with s = 1.0 still equals P2 (degenerate). -/
example :
    let (b0, b1, b2) := triBoundaries
    (triEvaluate b0 b1 b2 1.0 1.0 == ⟨0.5, 1.0, 0.0⟩) = true := by
  native_decide

end Curvenet
