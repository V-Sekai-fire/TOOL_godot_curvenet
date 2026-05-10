/-
MIT License — Copyright (c) 2026 K. S. Ernest (iFire) Lee.

End-to-end harmonic interpolation on a cut-mesh — slice 7 of the DeGoes22
rewrite.

This implements the scalar-valued analogue of DeGoes22 Eq. (6a):

  (VᵀLₕV) x_v = − Vᵀ Lₕ (C f_c)

where `f_c` are per-sample boundary values (one scalar per sample-side
column). The solution `x_v` is a vector of length nv whose unpromoted
mesh-vertex entries are the harmonic interpolation across the cut-mesh.
Promoted-vertex (sample) rows of `VᵀLₕV` are zero, so Gaussian elimination
returns 0 for them; the caller is expected to overlay sample values back
into those slots if needed.

The full DeGoes22 solve (Eq. 6a + Eq. 6b with 9-column deformation
gradients and 3-column vertex positions) repeats this scalar pipeline once
per right-hand-side column. Adding that loop is the next slice.
-/

import Curvenet.CutMesh
import Curvenet.CutMeshLaplacian
import Curvenet.DenseLinAlg
import Curvenet.PolygonLaplacian
import Curvenet.Vec3

namespace Curvenet
namespace HarmonicSolve

/-- Build the per-halfedge nh × k matrix from a per-sample nc × k sample
   matrix `Fc`: row `h` is `Fc[cColumnOf h]` if the halfedge has a sample
   target, else zeros. -/
def computeCFc (m : CutMesh) (sampleColumn : Nat → Nat → Bool → Nat)
    (Fc : DenseLinAlg.Mat) (k : Nat) : DenseLinAlg.Mat := Id.run do
  let nh := m.heCount
  let mut out : DenseLinAlg.Mat := DenseLinAlg.zeros nh k
  for h in [0:nh] do
    match m.cColumnOf h sampleColumn with
    | some c =>
        for j in [0:k] do
          out := DenseLinAlg.set out k h j (DenseLinAlg.get Fc k c j)
    | none => pure ()
  return out

/-- Multi-column scalar solve: simultaneously solve `(VᵀLₕV) X_v = −Vᵀ Lₕ (C F_c)`
   for k right-hand-side columns. k=9 covers DeGoes22 Eq. (6a) deformation
   gradients (flattened 3×3 per sample); k=3 covers Eq. (6b) vertex
   positions when the y_h transform term is folded in by the caller. -/
def solveMulti (m : CutMesh) (positions : Array Vec3)
    (sampleColumn : Nat → Nat → Bool → Nat)
    (Fc : DenseLinAlg.Mat) (k : Nat) : DenseLinAlg.Mat :=
  let nh := m.heCount
  let nv := m.vertexCount
  let Lh := CutMeshLaplacian.assembleLh m positions
  let V  := CutMeshLaplacian.assembleV m
  let Vt := DenseLinAlg.transpose nh nv V
  let LhV := DenseLinAlg.matMul nh nh nv Lh V
  let lhs := DenseLinAlg.matMul nv nh nv Vt LhV
  let CFc := computeCFc m sampleColumn Fc k
  let LhCFc := DenseLinAlg.matMul nh nh k Lh CFc
  let VtLhCFc := DenseLinAlg.matMul nv nh k Vt LhCFc
  -- RHS = − Vᵀ Lₕ C F_c
  let rhs : DenseLinAlg.Mat := VtLhCFc.map (fun x => -x)
  DenseLinAlg.solveMulti nv k lhs rhs

end HarmonicSolve

/- ============================================================ -/
/- Concrete end-to-end solve on triangleWithSample.            -/
/- ============================================================ -/

namespace HarmonicSolveExamples

open HarmonicSolve

/-- Equilateral triangle vertex positions. -/
private def triPositions : Array Vec3 :=
  #[ ⟨0.0, 0.0, 0.0⟩
   , ⟨1.0, 0.0, 0.0⟩
   , ⟨0.5, 0.0, 0.8660254037844386⟩
   ]

/-- One sample (vertex 0 promoted) → column 0. -/
private def oneSample : Nat → Nat → Bool → Nat := fun _ _ _ => 0

/- Multi-column solve checks: same problem but several RHS columns at once.
   `solveMulti` at k = 1 covers what the removed `solveScalar` used to
   prove; the cases below exercise k = 3 and k = 9. -/

/-- 3-column case (vertex positions). With sample value (5, 10, 15), the
   harmonic solve yields (5, 10, 15) at the unpromoted vertices. -/
example :
    -- Fc is 1 × 3 (one sample, three components).
    let Fc : DenseLinAlg.Mat := #[5.0, 10.0, 15.0]
    let Xv := solveMulti CutExamples.triangleWithSample triPositions oneSample Fc 3
    -- Xv is 3 × 3. Rows for vertices 1 and 2 should equal (5, 10, 15).
    let v1 := #[DenseLinAlg.get Xv 3 1 0, DenseLinAlg.get Xv 3 1 1, DenseLinAlg.get Xv 3 1 2]
    let v2 := #[DenseLinAlg.get Xv 3 2 0, DenseLinAlg.get Xv 3 2 1, DenseLinAlg.get Xv 3 2 2]
    DenseLinAlg.vecWithinEps v1 #[5.0, 10.0, 15.0] 1e-9 &&
    DenseLinAlg.vecWithinEps v2 #[5.0, 10.0, 15.0] 1e-9 = true := by
  native_decide

/-- 9-column case (flattened 3×3 deformation gradient). Sample carries the
   flattened identity matrix; harmonic solve propagates it to the
   unpromoted vertices. -/
example :
    -- Fc is 1 × 9 with the row-major identity matrix.
    let Fc : DenseLinAlg.Mat :=
      #[1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0]
    let Xv := solveMulti CutExamples.triangleWithSample triPositions oneSample Fc 9
    -- Vertex 1 (row 1 of Xv) should be the identity flattened.
    let v1 : Array Float := Array.ofFn (n := 9)
      (fun (j : Fin 9) => DenseLinAlg.get Xv 9 1 j.val)
    DenseLinAlg.vecWithinEps v1 #[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] 1e-9 = true := by
  native_decide

/-- Scalar (k=1) case: constant boundary value 7.0 at the sample propagates
   harmonically to a constant 7.0 at every unpromoted vertex. -/
example :
    let Fc : DenseLinAlg.Mat := #[7.0]
    let Xv := solveMulti CutExamples.triangleWithSample triPositions oneSample Fc 1
    let v1 := DenseLinAlg.get Xv 1 1 0
    let v2 := DenseLinAlg.get Xv 1 2 0
    Float.abs (v1 - 7.0) < 1e-9 && Float.abs (v2 - 7.0) < 1e-9 = true := by
  native_decide

/-- Asymmetric 3-column case: f_c = (1, 2, 3). Unpromoted vertices return
   (1, 2, 3) — independence of components, no cross-coupling. -/
example :
    let Fc : DenseLinAlg.Mat := #[1.0, 2.0, 3.0]
    let Xv := solveMulti CutExamples.triangleWithSample triPositions oneSample Fc 3
    let v1 := #[DenseLinAlg.get Xv 3 1 0, DenseLinAlg.get Xv 3 1 1, DenseLinAlg.get Xv 3 1 2]
    let v2 := #[DenseLinAlg.get Xv 3 2 0, DenseLinAlg.get Xv 3 2 1, DenseLinAlg.get Xv 3 2 2]
    DenseLinAlg.vecWithinEps v1 #[1.0, 2.0, 3.0] 1e-9 &&
    DenseLinAlg.vecWithinEps v2 #[1.0, 2.0, 3.0] 1e-9 = true := by
  native_decide

/-- 9-column rotated identity (R_z(90°) flattened) propagates as a rigid
   matrix value across the unpromoted vertices. -/
example :
    -- R_z(90°) row-major: ((0, -1, 0), (1, 0, 0), (0, 0, 1))
    let Fc : DenseLinAlg.Mat :=
      #[0.0, -1.0, 0.0,
        1.0,  0.0, 0.0,
        0.0,  0.0, 1.0]
    let Xv := solveMulti CutExamples.triangleWithSample triPositions oneSample Fc 9
    let v1 : Array Float := Array.ofFn (n := 9)
      (fun (j : Fin 9) => DenseLinAlg.get Xv 9 1 j.val)
    DenseLinAlg.vecWithinEps v1 #[0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0] 1e-9 = true := by
  native_decide

end HarmonicSolveExamples

end Curvenet
