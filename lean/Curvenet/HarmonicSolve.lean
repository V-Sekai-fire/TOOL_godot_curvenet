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

/-- Build C·f_c as a per-halfedge vector: each halfedge whose target is a
   curvenet sample picks up the sample's value; other halfedges are zero. -/
def computeCfc (m : CutMesh) (sampleColumn : Nat → Nat → Bool → Nat)
    (fc : Array Float) : Array Float := Id.run do
  let nh := m.heCount
  let mut out : Array Float := Array.replicate nh 0.0
  for h in [0:nh] do
    match m.cColumnOf h sampleColumn with
    | some c => out := out.set! h fc[c]!
    | none => pure ()
  return out

/-- Solve (VᵀLₕV) x_v = − Vᵀ Lₕ (C f_c) for x_v.

   `nc` is the total sample-column count (size of `fc`).
   `sampleColumn` packs (curveId, sampleIdx, side) → column index. -/
def solveScalar (m : CutMesh) (positions : Array Vec3)
    (sampleColumn : Nat → Nat → Bool → Nat)
    (fc : Array Float) : Array Float :=
  let nh := m.heCount
  let nv := m.vertexCount
  let Lh := CutMeshLaplacian.assembleLh m positions
  let V  := CutMeshLaplacian.assembleV m
  let Vt := DenseLinAlg.transpose nh nv V
  let LhV := DenseLinAlg.matMul nh nh nv Lh V
  let lhs := DenseLinAlg.matMul nv nh nv Vt LhV
  let cfc := computeCfc m sampleColumn fc
  let LhCfc := DenseLinAlg.matVec nh nh Lh cfc
  let VtLhCfc := DenseLinAlg.matVec nv nh Vt LhCfc
  -- RHS = − Vᵀ Lₕ C f_c
  let rhs : Array Float := VtLhCfc.map (fun x => -x)
  DenseLinAlg.solve nv lhs rhs

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

/-- Sample value 5.0 with a single sample column. -/
private def fc : Array Float := #[5.0]

/-- Trivial harmonic solve: with vertex 0 fixed to 5.0, the cot-Laplacian
   harmonic minimum on an equilateral triangle puts every other vertex
   at the same value (5.0). The promoted slot (index 0) returns 0 from
   the degenerate solve, but the *unpromoted* vertices 1 and 2 are
   harmonic-interpolated and equal 5.0. -/
example :
    let xv := solveScalar CutExamples.triangleWithSample triPositions oneSample fc
    ((xv[1]! - 5.0).abs < 1e-9 && (xv[2]! - 5.0).abs < 1e-9) = true := by
  native_decide

/-- The promoted-vertex slot is zeroed by Gaussian elimination on the
   degenerate row (it carries no harmonic energy in the system). -/
example :
    let xv := solveScalar CutExamples.triangleWithSample triPositions oneSample fc
    (xv[0]!.abs < 1e-12) = true := by
  native_decide

/-- A different sample value: setting fc=[2.5] should give vertices 1, 2
   the value 2.5 (linearity of the Laplacian solve). -/
example :
    let xv := solveScalar CutExamples.triangleWithSample triPositions oneSample #[2.5]
    ((xv[1]! - 2.5).abs < 1e-9 && (xv[2]! - 2.5).abs < 1e-9) = true := by
  native_decide

end HarmonicSolveExamples

end Curvenet
