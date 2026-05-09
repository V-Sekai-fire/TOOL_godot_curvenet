/-
MIT License — Copyright (c) 2026 K. S. Ernest (iFire) Lee.

Cut-mesh Laplacian assembly + VᵀLV matrix — slice 6 of the DeGoes22
rewrite.

DeGoes22 §4.2: Lₕ is the cut-mesh-wide Laplacian, gathered as the sum of
per-cut-face polygon Laplacians at the halfedge-corner indices belonging
to each face. V copies mesh-vertex unknowns to halfedges. The shared
LHS of the two solves in §4.3 is the nv × nv matrix VᵀLₕV.

For interior cut-faces (face = some f) we walk the face loop via `next`
pointers and collect the face-corner positions = `positions[target(h)]`
for each halfedge h in the loop. Boundary halfedges (face = none) don't
contribute to Lₕ. This matches the paper: only interior cut-faces carry
a polygon Laplacian; boundary halfedges enter the system through the
free-boundary condition implied by the Dirichlet energy minimization.
-/

import Curvenet.CutMesh
import Curvenet.DenseLinAlg
import Curvenet.PolygonLaplacian
import Curvenet.Vec3

namespace Curvenet
namespace CutMeshLaplacian

/-- Halfedges of cut-face `faceId`, in `next`-loop order. -/
def faceLoop (m : CutMesh) (faceId : Nat) : Array Nat := Id.run do
  let nh := m.heCount
  let mut startOpt : Option Nat := none
  for h in [0:nh] do
    if m.base.halfedges[h]!.face == some faceId then
      startOpt := some h
      break
  match startOpt with
  | none => return #[]
  | some start =>
    let mut acc : Array Nat := #[start]
    let mut cur := m.base.halfedges[start]!.next
    let mut steps : Nat := 0
    while cur ≠ start && steps < nh do
      acc := acc.push cur
      cur := m.base.halfedges[cur]!.next
      steps := steps + 1
    return acc

/-- Face-corner positions for cut-face `faceId`: `positions[target(h)]`
   walked in face-loop order. -/
def facePolygon (m : CutMesh) (positions : Array Vec3) (faceId : Nat) : Array Vec3 :=
  let halfedges := faceLoop m faceId
  halfedges.map fun h => positions[m.base.halfedges[h]!.target]!

/-- Assemble Lₕ : nh × nh by summing per-cut-face polygon Laplacians at
   the halfedge indices belonging to each face. -/
def assembleLh (m : CutMesh) (positions : Array Vec3) : DenseLinAlg.Mat := Id.run do
  let nh := m.heCount
  let mut L : DenseLinAlg.Mat := DenseLinAlg.zeros nh nh
  for f in [0:m.base.faceCount] do
    let halfedges := faceLoop m f
    let polygon := facePolygon m positions f
    let nf := polygon.size
    if nf < 3 then continue
    let Lf := PolygonLaplacian.polygonCotLaplacian polygon
    for li in [0:nf] do
      for lj in [0:nf] do
        let gi := halfedges[li]!
        let gj := halfedges[lj]!
        let cur := DenseLinAlg.get L nh gi gj
        L := DenseLinAlg.set L nh gi gj (cur + PolygonLaplacian.get Lf nf li lj)
  return L

/-- Assemble V : nh × nv. V[h, target(h)] = 1 when target is a mesh vertex,
   else the halfedge does not contribute to V (it goes through C instead). -/
def assembleV (m : CutMesh) : DenseLinAlg.Mat := Id.run do
  let nh := m.heCount
  let nv := m.vertexCount
  let mut V : DenseLinAlg.Mat := DenseLinAlg.zeros nh nv
  for h in [0:nh] do
    match m.vColumnOf h with
    | some col => V := DenseLinAlg.set V nv h col 1.0
    | none => pure ()
  return V

/-- The shared LHS of DeGoes22 Eq. (6): nv × nv. Symmetric, PSD when Lₕ is. -/
def assembleVtLhV (m : CutMesh) (positions : Array Vec3) : DenseLinAlg.Mat :=
  let nh := m.heCount
  let nv := m.vertexCount
  let L  := assembleLh m positions
  let V  := assembleV m
  let Vt := DenseLinAlg.transpose nh nv V
  let LV := DenseLinAlg.matMul nh nh nv L V
  DenseLinAlg.matMul nv nh nv Vt LV

end CutMeshLaplacian

/- ============================================================ -/
/- Concrete instance checks on the uncut triangle.             -/
/- ============================================================ -/

namespace CutMeshLaplacianExamples

open CutMeshLaplacian

/-- Equilateral triangle vertex positions (matches slice 2's example). -/
private def triPositions : Array Vec3 :=
  #[ ⟨0.0, 0.0, 0.0⟩
   , ⟨1.0, 0.0, 0.0⟩
   , ⟨0.5, 0.0, 0.8660254037844386⟩
   ]

/-- Unit square (CCW). -/
private def quadPositions : Array Vec3 :=
  #[ ⟨0.0, 0.0, 0.0⟩
   , ⟨1.0, 0.0, 0.0⟩
   , ⟨1.0, 0.0, 1.0⟩
   , ⟨0.0, 0.0, 1.0⟩
   ]

/-- Face loop on the uncut triangle has 3 halfedges (the interior face). -/
example : (faceLoop CutExamples.uncutTriangle 0).size = 3 := by native_decide

/-- VᵀLₕV on uncut triangle is 3×3. -/
example : (assembleVtLhV CutExamples.uncutTriangle triPositions).size = 9 := by native_decide

/-- VᵀLₕV row sums vanish (kernel contains the constant vector). -/
example :
    let M := assembleVtLhV CutExamples.uncutTriangle triPositions
    PolygonLaplacian.rowSumsWithin M 3 1e-12 = true := by native_decide

/-- VᵀLₕV is symmetric. -/
example :
    let M := assembleVtLhV CutExamples.uncutTriangle triPositions
    PolygonLaplacian.isSymmetricWithin M 3 1e-12 = true := by native_decide

/-- For the uncut quad, VᵀLₕV is 4×4. -/
example : (assembleVtLhV CutExamples.uncutQuad quadPositions).size = 16 := by native_decide

/-- Quad VᵀLₕV row sums vanish. -/
example :
    let M := assembleVtLhV CutExamples.uncutQuad quadPositions
    PolygonLaplacian.rowSumsWithin M 4 1e-12 = true := by native_decide

/-- Quad VᵀLₕV is symmetric. -/
example :
    let M := assembleVtLhV CutExamples.uncutQuad quadPositions
    PolygonLaplacian.isSymmetricWithin M 4 1e-12 = true := by native_decide

end CutMeshLaplacianExamples

end Curvenet
