/-
MIT License — Copyright (c) 2026 K. S. Ernest (iFire) Lee.

Bilinear-blended Coons patch over four cubic-Bezier boundary curves,
mirroring src/curvenet/coons_patch.h.
-/

import Curvenet.Vec3
import Curvenet.Bezier

namespace Curvenet

/-- One side of a Coons patch: a cubic Bezier with 4 control points. -/
structure BoundaryCurve where
  c0 : Vec3
  c1 : Vec3
  c2 : Vec3
  c3 : Vec3

@[inline] def BoundaryCurve.beq (a b : BoundaryCurve) : Bool :=
  a.c0 == b.c0 && a.c1 == b.c1 && a.c2 == b.c2 && a.c3 == b.c3

instance : BEq BoundaryCurve := ⟨BoundaryCurve.beq⟩

/-- Evaluate a boundary curve at parameter s ∈ [0,1]. -/
def BoundaryCurve.evaluate (b : BoundaryCurve) (s : Float) : Vec3 :=
  bezier b.c0 b.c1 b.c2 b.c3 s

/-- Bilinear-blended Coons patch.

   * `u0(u)` is the bottom edge at v = 0;
   * `u1(u)` is the top edge at v = 1;
   * `v0(v)` is the left edge at u = 0;
   * `v1(v)` is the right edge at u = 1.

   Corner compatibility: u0.c0 = v0.c0, u0.c3 = v1.c0, u1.c0 = v0.c3, u1.c3 = v1.c3. -/
structure CoonsPatch where
  u0 : BoundaryCurve
  u1 : BoundaryCurve
  v0 : BoundaryCurve
  v1 : BoundaryCurve

/-- Coons evaluation: S(u,v) = Lc + Ld − B, where Lc/Ld are ruled in v/u and
    B is the bilinear blend of corners. -/
def CoonsPatch.evaluate (p : CoonsPatch) (u v : Float) : Vec3 :=
  let mu := 1.0 - u
  let mv := 1.0 - v
  let cu0 := p.u0.evaluate u
  let cu1 := p.u1.evaluate u
  let cv0 := p.v0.evaluate v
  let cv1 := p.v1.evaluate v
  let Lc  := mv * cu0 + v * cu1
  let Ld  := mu * cv0 + u * cv1
  let P00 := p.u0.c0
  let P10 := p.u0.c3
  let P01 := p.u1.c0
  let P11 := p.u1.c3
  let B := (mu * mv) * P00 + (u * mv) * P10 + (mu * v) * P01 + (u * v) * P11
  Lc + Ld - B

/- ============================================================ -/
/- Concrete corner-recovery checks on a flat unit-square patch. -/
/- ============================================================ -/

private def lerp3 (a b : Vec3) (t : Float) : Vec3 :=
  (1.0 - t) * a + t * b

/-- A corner-consistent flat unit-square Coons patch with linear boundaries. -/
def flatUnitPatch : CoonsPatch :=
  let P00 : Vec3 := ⟨0.0, 0.0, 0.0⟩
  let P10 : Vec3 := ⟨1.0, 0.0, 0.0⟩
  let P11 : Vec3 := ⟨1.0, 1.0, 0.0⟩
  let P01 : Vec3 := ⟨0.0, 1.0, 0.0⟩
  { u0 := { c0 := P00, c1 := lerp3 P00 P10 (1.0/3.0), c2 := lerp3 P00 P10 (2.0/3.0), c3 := P10 }
  , u1 := { c0 := P01, c1 := lerp3 P01 P11 (1.0/3.0), c2 := lerp3 P01 P11 (2.0/3.0), c3 := P11 }
  , v0 := { c0 := P00, c1 := lerp3 P00 P01 (1.0/3.0), c2 := lerp3 P00 P01 (2.0/3.0), c3 := P01 }
  , v1 := { c0 := P10, c1 := lerp3 P10 P11 (1.0/3.0), c2 := lerp3 P10 P11 (2.0/3.0), c3 := P11 }
  }

/-- Corner (0,0) interpolates P00. -/
example : (flatUnitPatch.evaluate 0.0 0.0 == ⟨0.0, 0.0, 0.0⟩) = true := by native_decide

/-- Corner (1,0) interpolates P10. -/
example : (flatUnitPatch.evaluate 1.0 0.0 == ⟨1.0, 0.0, 0.0⟩) = true := by native_decide

/-- Corner (1,1) interpolates P11. -/
example : (flatUnitPatch.evaluate 1.0 1.0 == ⟨1.0, 1.0, 0.0⟩) = true := by native_decide

/-- Corner (0,1) interpolates P01. -/
example : (flatUnitPatch.evaluate 0.0 1.0 == ⟨0.0, 1.0, 0.0⟩) = true := by native_decide

/-- Patch center is at the unit-square centroid. -/
example : (flatUnitPatch.evaluate 0.5 0.5 == ⟨0.5, 0.5, 0.0⟩) = true := by native_decide

end Curvenet
