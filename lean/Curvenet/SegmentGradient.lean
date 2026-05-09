/-
MIT License — Copyright (c) 2026 K. S. Ernest (iFire) Lee.

Per-segment deformation gradient dispatcher — slice 18 of the DeGoes22
rewrite.

A curvenet segment can sit in three places along a curve, and the
paper specifies a different `F_i` formula for each:

  isolated     — curve has no intersections (closed loop or anchor-anchor):
                 `F = (l/l̆) · R(t̆, t)`              (slice 4)
  intersection — segment whose endpoint(s) are at an intersection:
                 `F^± = (B^± S^±)·(B̆^± S̆^±)⁻¹`     (slices 13–15)
  midCurve     — non-intersection segment on a curve that has
                 intersections at its endpoints. The per-side scaled
                 frames are interpolated via slice 16, then the same
                 intersection formula `F^± = (B^± S^±)·(B̆^± S̆^±)⁻¹`
                 applies.

This file exposes the dispatcher so that the §4.3 constraint matrix
`f_c` can call a single function per segment side without picking the
formula at the call site.
-/

import Curvenet.ScaledFrames
import Curvenet.Vec3

namespace Curvenet
namespace SegmentGradient

open ScaledFrames

/-- Isolated-curve deformation gradient. For a segment running
   `restStart → restEnd` in the rest pose and `posedStart → posedEnd`
   in the posed pose, returns the single 3×3 matrix
   `F = (l/l̆) · R(t̆, t)`. Both sides of the segment receive the same
   gradient when the curve has no intersections. -/
def isolated (restStart restEnd posedStart posedEnd : Vec3) : Mat3 :=
  isolatedSegmentGradient restStart restEnd posedStart posedEnd

/-- Intersection-adjacent (or mid-curve) per-side deformation gradients.
   Inputs are the rest and posed scaled-frame matrices for each side.
   Returns the pair `(F^+, F^-) = ((B^+ S^+)·(B̆^+ S̆^+)⁻¹,
                                    (B^- S^-)·(B̆^- S̆^-)⁻¹)`. -/
def intersectionPair
    (restPlus  posedPlus  : Mat3)
    (restMinus posedMinus : Mat3) : Mat3 × Mat3 :=
  ( deformationGradient restPlus  posedPlus
  , deformationGradient restMinus posedMinus )

/-- Wraps the two cases under a tagged input so callers don't have to
   pick the formula. The midCurve case routes through the same
   `intersectionPair` formula because slice 16's interpolation produces
   per-segment-side `(B S)` matrices that the slice 15 inverse handles
   identically. -/
inductive SegmentInputs
  | isolated     (restStart restEnd posedStart posedEnd : Vec3)
  | intersection (restPlus posedPlus restMinus posedMinus : Mat3)

/-- Compute the per-side deformation gradient pair for any segment.
   For an isolated curve both sides receive the same matrix; for
   intersection-adjacent or mid-curve segments the sides can differ. -/
def gradientPair : SegmentInputs → Mat3 × Mat3
  | SegmentInputs.isolated rs re ps pe =>
      let F := isolated rs re ps pe
      (F, F)
  | SegmentInputs.intersection rp pp rm pm =>
      intersectionPair rp pp rm pm

end SegmentGradient

/- ============================================================ -/
/- Dispatcher invariants on tiny instances.                    -/
/- ============================================================ -/

namespace SegmentGradientExamples

open ScaledFrames
open SegmentGradient

private def origin : Vec3 := ⟨0.0, 0.0, 0.0⟩
private def xUnit  : Vec3 := ⟨1.0, 0.0, 0.0⟩

/-- Isolated branch, rest = posed → F = I on both sides. -/
example :
    let (Fp, Fm) := gradientPair (.isolated origin xUnit origin xUnit)
    mat3WithinEps Fp mat3Identity 1e-12 ∧
    mat3WithinEps Fm mat3Identity 1e-12 := by
  native_decide

/-- Isolated branch, rotated by 90°: F is the z-rotation on both sides. -/
example :
    let R90 : Mat3 :=
      mat3Mk 0.0 (-1.0) 0.0  1.0 0.0 0.0  0.0 0.0 1.0
    let (Fp, Fm) :=
      gradientPair (.isolated origin xUnit origin ⟨0.0, 1.0, 0.0⟩)
    mat3WithinEps Fp R90 1e-12 ∧ mat3WithinEps Fm R90 1e-12 := by
  native_decide

/-- Intersection branch, both sides identical rest = posed → F = I per side. -/
example :
    let frame : Mat3 :=
      mat3Mk 1.0 0.0 0.0  0.0 1.0 0.0  0.0 0.0 1.0
    let (Fp, Fm) :=
      gradientPair (.intersection frame frame frame frame)
    mat3WithinEps Fp mat3Identity 1e-12 ∧
    mat3WithinEps Fm mat3Identity 1e-12 := by
  native_decide

/-- Intersection branch with different rest/posed scales per side:
   plus side scales by 2 in width, minus side stays identity. -/
example :
    let restPlus  : Mat3 :=
      mat3Mk 1.0 0.0 0.0  0.0 1.0 0.0  0.0 0.0 1.0
    let posedPlus : Mat3 :=
      mat3Mk 1.0 0.0 0.0  0.0 2.0 0.0  0.0 0.0 1.0
    let restMinus  : Mat3 :=
      mat3Mk 1.0 0.0 0.0  0.0 1.0 0.0  0.0 0.0 1.0
    let posedMinus : Mat3 := restMinus
    let (Fp, Fm) :=
      gradientPair (.intersection restPlus posedPlus restMinus posedMinus)
    -- F^+ should be diag(1, 2, 1); F^- should be identity.
    let expectedFp : Mat3 :=
      mat3Mk 1.0 0.0 0.0  0.0 2.0 0.0  0.0 0.0 1.0
    mat3WithinEps Fp expectedFp 1e-12 ∧
    mat3WithinEps Fm mat3Identity 1e-12 := by
  native_decide

end SegmentGradientExamples

end Curvenet
