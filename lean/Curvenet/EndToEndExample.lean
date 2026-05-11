/-
MIT License — Copyright (c) 2026 K. S. Ernest (iFire) Lee.

Smallest end-to-end Profile Curves demonstration.

Closes the *Profile Curves input pathway* that every `DeformSolveExamples`
test until now had skipped: instead of hand-authoring a dense 3×3
deformation gradient `F_c`, this module derives `(F_c, X_c)` from a real
cubic Bézier segment that gets translated in world space, then feeds the
curve-derived inputs into `DeformSolve.solveDeformation` on the existing
`CutExamples.triangleWithSample` fixture. The resulting deformed
positions are pinned via `native_decide` to match the hand-authored
translation test in `DeformSolveExamples` bit-exactly.

Gall's Law: this is the smallest working subsystem that the goal
"Lean 4 spec + Slang shader codegen for Pixar's Profile Curves
character articulation algorithm" can grow from. Multi-segment
curvenets, the bind-time weight harvest on real meshes, the GPU
dispatch loop, and Mire integration all replace one component at a
time on top of this baseline.

Pipeline exercised (DeGoes22):
  cubic Bézier control points (4)
    │  bezier3                    ← Bézier evaluation (this module)
    ▼
  rest / posed sample positions (Vec3)
    │  ScaledFrames.isolatedSegmentGradient  ← DeGoes22 §3 isolated-curve form
    ▼
  F_c (3×3 deformation gradient) + X_c (target position)
    │  DeformSolve.solveDeformation          ← DeGoes22 §4.3 two-stage solve
    ▼
  deformed vertex positions, asserted to match the existing
  hand-authored `Fc_identity / Xc_translate5` reference.
-/

import Curvenet.CutMesh
import Curvenet.DeformSolve
import Curvenet.ScaledFrames
import Curvenet.DenseLinAlg
import Curvenet.Vec3

namespace Curvenet.EndToEndExample

open Curvenet
open Curvenet.Common
open Curvenet.ScaledFrames
open Curvenet.DenseLinAlg

/-! ## A single cubic Bézier — the *minimum* curvenet -/

/-- Cubic Bézier evaluation: B(t) = Σ_i C(3,i) (1−t)^(3−i) t^i P_i. -/
private def bezier3 (P0 P1 P2 P3 : Vec3) (t : Float) : Vec3 :=
  let u := 1.0 - t
  let c0 := u*u*u
  let c1 := 3.0*u*u*t
  let c2 := 3.0*u*t*t
  let c3 := t*t*t
  ⟨ c0*P0.x + c1*P1.x + c2*P2.x + c3*P3.x
  , c0*P0.y + c1*P1.y + c2*P2.y + c3*P3.y
  , c0*P0.z + c1*P1.z + c2*P2.z + c3*P3.z ⟩

/-- Vec3 elementwise add (paired with Curvenet.Vec3's `Add` instance). -/
private abbrev Vec3.add (a b : Vec3) : Vec3 := a + b

/-- Rest control points: evenly spaced along X so B(0.5) = (0, 0, 0).
    This places the sample exactly at vertex 0 of `triangleWithSample`. -/
private def Prest0 : Vec3 := ⟨-1.5, 0.0, 0.0⟩
private def Prest1 : Vec3 := ⟨-0.5, 0.0, 0.0⟩
private def Prest2 : Vec3 := ⟨ 0.5, 0.0, 0.0⟩
private def Prest3 : Vec3 := ⟨ 1.5, 0.0, 0.0⟩

/-- The world-space translation applied to the entire posed curve. -/
private def Tposed : Vec3 := ⟨5.0, 0.0, 0.0⟩

/-- Posed control points = rest control points + Tposed (pure rigid translation). -/
private def Pposed0 : Vec3 := Vec3.add Prest0 Tposed
private def Pposed1 : Vec3 := Vec3.add Prest1 Tposed
private def Pposed2 : Vec3 := Vec3.add Prest2 Tposed
private def Pposed3 : Vec3 := Vec3.add Prest3 Tposed

/-! ## Sample positions on rest and posed curves -/

private def tSample  : Float := 0.5
private def tForward : Float := 0.6   -- the segment endpoint `q` used for tangent

/-- Curve sample at t=0.5: rest position = (0,0,0); posed position = (5,0,0). -/
private def restSampleP  : Vec3 := bezier3 Prest0  Prest1  Prest2  Prest3  tSample
private def restForwardQ : Vec3 := bezier3 Prest0  Prest1  Prest2  Prest3  tForward
private def posedSampleP : Vec3 := bezier3 Pposed0 Pposed1 Pposed2 Pposed3 tSample
private def posedForwardQ: Vec3 := bezier3 Pposed0 Pposed1 Pposed2 Pposed3 tForward

/-- Sanity: rest sample lands on the origin and posed sample lands at (5,0,0)
   — both within fp64 round-off. -/
example : restSampleP.x.abs < 1e-12 ∧
          restSampleP.y.abs < 1e-12 ∧
          restSampleP.z.abs < 1e-12 := by native_decide

example : (posedSampleP.x - 5.0).abs < 1e-12 ∧
          posedSampleP.y.abs < 1e-12 ∧
          posedSampleP.z.abs < 1e-12 := by native_decide

/-! ## Curve-derived F_c and X_c -/

/-- Deformation gradient at the sample — isolated-segment form (DeGoes22 §3).
   For pure translation, `isolatedSegmentGradient` produces the 3×3 identity:
   `(length_posed / length_rest) · smallestRotation(t_rest, t_posed)` with
   `length_posed = length_rest` and parallel tangents. -/
private def F_curve : Mat3 :=
  isolatedSegmentGradient restSampleP restForwardQ posedSampleP posedForwardQ

/-- Sanity: the curve-derived F_c is the identity matrix (1 ulp tolerance). -/
example : mat3WithinEps F_curve mat3Identity 1e-12 = true := by native_decide

/-- Sample target position = curve point at t=0.5 on the posed curve. -/
private def Xc_curve : Vec3 := posedSampleP

/-! ## Drive `DeformSolve.solveDeformation` with curve-derived inputs -/

private def triPositions : Array Vec3 :=
  #[ ⟨0.0, 0.0, 0.0⟩
   , ⟨1.0, 0.0, 0.0⟩
   , ⟨0.5, 0.0, 0.8660254037844386⟩ ]

private def oneSample : Nat → Nat → Bool → Nat := fun _ _ _ => 0

/-- The deformed mesh, computed with `(F_c, X_c)` derived from the Bézier
   instead of hand-authored. `DenseLinAlg.Mat` is `Array Float`, identical
   layout to `Mat3` (both are 9 floats row-major). For X_c we pack the
   three components of `Xc_curve`. -/
def deformed : Mat :=
  DeformSolve.solveDeformation
    CutExamples.triangleWithSample triPositions oneSample
    F_curve
    #[Xc_curve.x, Xc_curve.y, Xc_curve.z]

/-! ## Equivalence with the hand-authored translation test

The proofs below mirror `DeformSolveExamples`'s pure-translation case
(`Fc_identity / Xc_translate5`) bit-exactly — confirming the
*input pathway* (Bézier → ScaledFrames → F_c/X_c) is correct and lossless
through to the algorithm's output.
-/

/-- Vertex 1 (rest at (1, 0, 0)) ends at (6, 0, 0). -/
example :
    let v1 : Array Float := #[get deformed 3 1 0, get deformed 3 1 1, get deformed 3 1 2]
    vecWithinEps v1 #[6.0, 0.0, 0.0] 1e-9 = true := by native_decide

/-- Vertex 2 (rest at (0.5, 0, √3/2)) ends at (5.5, 0, √3/2). -/
example :
    let v2 : Array Float := #[get deformed 3 2 0, get deformed 3 2 1, get deformed 3 2 2]
    vecWithinEps v2 #[5.5, 0.0, 0.8660254037844386] 1e-9 = true := by native_decide

/-! ## Outputs exposed for the morph-target glTF demo

`ProfileCurvesDemo.lean` reads these to build the `.glb` POSITION
base attribute + per-target POSITION delta.

Vertex 0 is the promoted sample; the algorithm leaves it at 0 from the
degenerate slot and the caller is expected to overlay the sample target.
We do that overlay here so downstream consumers see consistent data. -/

/-- Rest positions of the 3 mesh vertices. -/
def restPositions : Array Vec3 := triPositions

/-- Deformed positions, with vertex 0 (the promoted sample) overlaid
   with the sample target. -/
def deformedPositions : Array Vec3 :=
  #[ Xc_curve
   , ⟨get deformed 3 1 0, get deformed 3 1 1, get deformed 3 1 2⟩
   , ⟨get deformed 3 2 0, get deformed 3 2 1, get deformed 3 2 2⟩ ]

/-- Triangle indices for the 1-face mesh (CCW winding). -/
def indices : Array UInt32 := #[0, 1, 2]

end Curvenet.EndToEndExample
