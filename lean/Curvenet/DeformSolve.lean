/-
MIT License — Copyright (c) 2026 K. S. Ernest (iFire) Lee.

Full DeGoes22 deformation solve — slice 9 of the rewrite.

Wires together both stages of DeGoes22 §4.3 Eq. (6):

  Stage 1 (Eq. 6a)   (VᵀLₕV) f_v = − Vᵀ Lₕ (C f_c)
  Stage 2 (Eq. 6b)   (VᵀLₕV) x_v = − Vᵀ Lₕ (C x_c − y_h)

with the bridge from stage 1 to stage 2 going through Eq. (3):

  f_h = V f_v + C f_c

then averaged over each cut-face's halfedges to obtain F_f, then applied
to rest face polygons to produce the per-halfedge target positions

  y_h[h] = X̆_f F_f^T   for the halfedge h that lives in cut-face f.

Tested end-to-end on `triangleWithSample` with f_c = identity (no local
rotation/stretch) and x_c = (5, 0, 0) (translate the sample by 5 in x).
The harmonic solve produces pure translation: every unpromoted vertex
moves to rest_position + (5, 0, 0).
-/

import Curvenet.CutMesh
import Curvenet.CutMeshLaplacian
import Curvenet.DenseLinAlg
import Curvenet.HarmonicSolve
import Curvenet.Vec3

namespace Curvenet
namespace DeformSolve

/-- Eq. (3): f_h = V f_v + C f_c. For each halfedge h:
   * if target(h) is a mesh-vertex, f_h[h, :] = f_v[target(h), :]
   * if target(h) is a sample,      f_h[h, :] = f_c[sample_col, :]
   * otherwise (edge intersection)  f_h[h, :] = 0
-/
def computeFh (m : CutMesh) (sampleColumn : Nat → Nat → Bool → Nat)
    (Fv : DenseLinAlg.Mat) (Fc : DenseLinAlg.Mat) (k : Nat) : DenseLinAlg.Mat := Id.run do
  let nh := m.heCount
  let mut Fh : DenseLinAlg.Mat := DenseLinAlg.zeros nh k
  for h in [0:nh] do
    match m.vColumnOf h, m.cColumnOf h sampleColumn with
    | some vCol, _ =>
        for j in [0:k] do
          Fh := DenseLinAlg.set Fh k h j (DenseLinAlg.get Fv k vCol j)
    | _, some cCol =>
        for j in [0:k] do
          Fh := DenseLinAlg.set Fh k h j (DenseLinAlg.get Fc k cCol j)
    | _, _ => pure ()
  return Fh

/-- Average f_h over each cut-face's halfedges to obtain a per-face F_f.
   Output is faceCount × k. -/
def averageOverFaces (m : CutMesh) (Fh : DenseLinAlg.Mat) (k : Nat) : DenseLinAlg.Mat := Id.run do
  let nFaces := m.base.faceCount
  let mut Ff : DenseLinAlg.Mat := DenseLinAlg.zeros nFaces k
  for f in [0:nFaces] do
    let halfedges := CutMeshLaplacian.faceLoop m f
    let nf := halfedges.size
    if nf == 0 then continue
    let invN := 1.0 / Float.ofNat nf
    for j in [0:k] do
      let mut s : Float := 0.0
      for i in [0:nf] do
        s := s + DenseLinAlg.get Fh k halfedges[i]! j
      Ff := DenseLinAlg.set Ff k f j (s * invN)
  return Ff

/-- y_h[h] = positions[target(h)] @ F_f^T (using the F_f for h's face).
   Boundary halfedges (face = none) yield zeros. F_f stored row-major as
   9-element row of `Ff`. -/
def computeYh (m : CutMesh) (positions : Array Vec3) (Ff : DenseLinAlg.Mat) : DenseLinAlg.Mat := Id.run do
  let nh := m.heCount
  let mut yh : DenseLinAlg.Mat := DenseLinAlg.zeros nh 3
  for h in [0:nh] do
    let he := m.base.halfedges[h]!
    match he.face with
    | some f =>
        let pos := positions[he.target]!
        let f00 := DenseLinAlg.get Ff 9 f 0
        let f01 := DenseLinAlg.get Ff 9 f 1
        let f02 := DenseLinAlg.get Ff 9 f 2
        let f10 := DenseLinAlg.get Ff 9 f 3
        let f11 := DenseLinAlg.get Ff 9 f 4
        let f12 := DenseLinAlg.get Ff 9 f 5
        let f20 := DenseLinAlg.get Ff 9 f 6
        let f21 := DenseLinAlg.get Ff 9 f 7
        let f22 := DenseLinAlg.get Ff 9 f 8
        -- (X̆_f F_f^T)[i, j] = pos · F_f[j, :]
        yh := DenseLinAlg.set yh 3 h 0 (pos.x * f00 + pos.y * f01 + pos.z * f02)
        yh := DenseLinAlg.set yh 3 h 1 (pos.x * f10 + pos.y * f11 + pos.z * f12)
        yh := DenseLinAlg.set yh 3 h 2 (pos.x * f20 + pos.y * f21 + pos.z * f22)
    | none => pure ()
  return yh

/-- Full DeGoes22 deformation solve.

   Inputs:
   * `m`             — cut-mesh with promoted-sample vertex kinds
   * `positions`     — rest 3D positions, indexed by `m.vertexCount`
   * `sampleColumn`  — packs (curveId, sampleIdx, side) → C-column index
   * `Fc`            — nc × 9 sample deformation gradients (flattened 3×3)
   * `Xc`            — nc × 3  sample target positions

   Output: nv × 3 deformed vertex positions (promoted slots return 0 from
   the degenerate solve; the caller can overlay sample positions there). -/
def solveDeformation (m : CutMesh) (positions : Array Vec3)
    (sampleColumn : Nat → Nat → Bool → Nat)
    (Fc : DenseLinAlg.Mat) (Xc : DenseLinAlg.Mat) : DenseLinAlg.Mat :=
  let nh := m.heCount
  let nv := m.vertexCount
  -- Stage 1 (Eq. 6a): solve for f_v.
  let Fv := HarmonicSolve.solveMulti m positions sampleColumn Fc 9
  -- Bridge (Eq. 3 + per-face average): build F_f.
  let Fh := computeFh m sampleColumn Fv Fc 9
  let Ff := averageOverFaces m Fh 9
  -- y_h from rest polygons + F_f.
  let yh := computeYh m positions Ff
  -- Stage 2 (Eq. 6b): solve for x_v.
  let Lh := CutMeshLaplacian.assembleLh m positions
  let V  := CutMeshLaplacian.assembleV m
  let Vt := DenseLinAlg.transpose nh nv V
  let CXc := HarmonicSolve.computeCFc m sampleColumn Xc 3
  -- diff = C X_c − y_h  (nh × 3)
  let diff : DenseLinAlg.Mat :=
    Array.ofFn (n := nh * 3) (fun (k : Fin (nh * 3)) => CXc[k.val]! - yh[k.val]!)
  let LhDiff := DenseLinAlg.matMul nh nh 3 Lh diff
  let VtLhDiff := DenseLinAlg.matMul nv nh 3 Vt LhDiff
  let rhs : DenseLinAlg.Mat := VtLhDiff.map (fun x => -x)
  let LhV := DenseLinAlg.matMul nh nh nv Lh V
  let lhs := DenseLinAlg.matMul nv nh nv Vt LhV
  DenseLinAlg.solveMulti nv 3 lhs rhs

end DeformSolve

/- ============================================================ -/
/- End-to-end deformation: pure translation by (5, 0, 0) on    -/
/- triangleWithSample, with f_c = identity.                    -/
/- ============================================================ -/

namespace DeformSolveExamples

open DeformSolve

private def triPositions : Array Vec3 :=
  #[ ⟨0.0, 0.0, 0.0⟩
   , ⟨1.0, 0.0, 0.0⟩
   , ⟨0.5, 0.0, 0.8660254037844386⟩
   ]

private def oneSample : Nat → Nat → Bool → Nat := fun _ _ _ => 0

/-- Sample carries identity deformation gradient (no local rotation /
   stretch) and target position (5, 0, 0). -/
private def Fc_identity : DenseLinAlg.Mat :=
  #[1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0]

private def Xc_translate5 : DenseLinAlg.Mat := #[5.0, 0.0, 0.0]

/-- The full solve produces pure translation: vertex 1 (rest at (1, 0, 0))
   ends up at (6, 0, 0). -/
example :
    let Xv := solveDeformation CutExamples.triangleWithSample triPositions oneSample
                Fc_identity Xc_translate5
    let v1 : Array Float := #[DenseLinAlg.get Xv 3 1 0,
                               DenseLinAlg.get Xv 3 1 1,
                               DenseLinAlg.get Xv 3 1 2]
    DenseLinAlg.vecWithinEps v1 #[6.0, 0.0, 0.0] 1e-9 = true := by
  native_decide

/-- Vertex 2 (rest at (0.5, 0, √3/2)) ends up at (5.5, 0, √3/2). -/
example :
    let Xv := solveDeformation CutExamples.triangleWithSample triPositions oneSample
                Fc_identity Xc_translate5
    let v2 : Array Float := #[DenseLinAlg.get Xv 3 2 0,
                               DenseLinAlg.get Xv 3 2 1,
                               DenseLinAlg.get Xv 3 2 2]
    DenseLinAlg.vecWithinEps v2 #[5.5, 0.0, 0.8660254037844386] 1e-9 = true := by
  native_decide

/-- A second translation: x_c = (0, 2, 0) translates the y-component by 2. -/
example :
    let Xc : DenseLinAlg.Mat := #[0.0, 2.0, 0.0]
    let Xv := solveDeformation CutExamples.triangleWithSample triPositions oneSample
                Fc_identity Xc
    let v1 : Array Float := #[DenseLinAlg.get Xv 3 1 0,
                               DenseLinAlg.get Xv 3 1 1,
                               DenseLinAlg.get Xv 3 1 2]
    DenseLinAlg.vecWithinEps v1 #[1.0, 2.0, 0.0] 1e-9 = true := by
  native_decide

/-- Pure rotation R_z(90°) applied at the sample (vertex 0 at origin
   stays at origin under this rotation). The deformation gradient
   propagates harmonically as the same rotation matrix; vertex 1
   (rest at (1, 0, 0)) ends at R_z(90°)·(1, 0, 0) = (0, 1, 0). -/
private def Fc_rotZ90 : DenseLinAlg.Mat :=
  #[0.0, -1.0, 0.0,
    1.0,  0.0, 0.0,
    0.0,  0.0, 1.0]

private def Xc_origin : DenseLinAlg.Mat := #[0.0, 0.0, 0.0]

example :
    let Xv := solveDeformation CutExamples.triangleWithSample triPositions oneSample
                Fc_rotZ90 Xc_origin
    let v1 : Array Float := #[DenseLinAlg.get Xv 3 1 0,
                               DenseLinAlg.get Xv 3 1 1,
                               DenseLinAlg.get Xv 3 1 2]
    -- (1, 0, 0) rotated 90° about z = (0, 1, 0)
    DenseLinAlg.vecWithinEps v1 #[0.0, 1.0, 0.0] 1e-9 = true := by
  native_decide

example :
    let Xv := solveDeformation CutExamples.triangleWithSample triPositions oneSample
                Fc_rotZ90 Xc_origin
    let v2 : Array Float := #[DenseLinAlg.get Xv 3 2 0,
                               DenseLinAlg.get Xv 3 2 1,
                               DenseLinAlg.get Xv 3 2 2]
    -- (0.5, 0, √3/2) rotated 90° about z = (0, 0.5, √3/2)
    DenseLinAlg.vecWithinEps v2 #[0.0, 0.5, 0.8660254037844386] 1e-9 = true := by
  native_decide

/-- Uniform scale by 2 with the sample fixed at origin: every unpromoted
   vertex's distance from the origin doubles. -/
private def Fc_scale2 : DenseLinAlg.Mat :=
  #[2.0, 0.0, 0.0,
    0.0, 2.0, 0.0,
    0.0, 0.0, 2.0]

example :
    let Xv := solveDeformation CutExamples.triangleWithSample triPositions oneSample
                Fc_scale2 Xc_origin
    let v1 : Array Float := #[DenseLinAlg.get Xv 3 1 0,
                               DenseLinAlg.get Xv 3 1 1,
                               DenseLinAlg.get Xv 3 1 2]
    -- (1, 0, 0) scaled 2× = (2, 0, 0)
    DenseLinAlg.vecWithinEps v1 #[2.0, 0.0, 0.0] 1e-9 = true := by
  native_decide

example :
    let Xv := solveDeformation CutExamples.triangleWithSample triPositions oneSample
                Fc_scale2 Xc_origin
    let v2 : Array Float := #[DenseLinAlg.get Xv 3 2 0,
                               DenseLinAlg.get Xv 3 2 1,
                               DenseLinAlg.get Xv 3 2 2]
    -- (0.5, 0, √3/2) scaled 2× = (1, 0, √3)
    DenseLinAlg.vecWithinEps v2 #[1.0, 0.0, 1.7320508075688772] 1e-9 = true := by
  native_decide

end DeformSolveExamples

end Curvenet
