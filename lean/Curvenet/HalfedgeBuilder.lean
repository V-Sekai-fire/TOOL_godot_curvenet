/-
MIT License — Copyright (c) 2026 K. S. Ernest (iFire) Lee.

Triangle-soup → HalfedgeMesh builder. Mirrors `src/curvenet/halfedge_builder.h`.

Conventions (matching the C++ side):
  * Triangle f has CCW vertices `(t0, t1, t2)`.
  * Interior halfedges: `3f` is `t0 → t1`, `3f+1` is `t1 → t2`, `3f+2` is `t2 → t0`.
  * Boundary halfedges follow the interior block, one per interior orphan.
  * Each boundary halfedge has the opposite direction of its interior twin
    and `face = none`.

The Lean version stays algorithmically identical to the C++ so the same
small-instance manifold checks (single triangle, two-triangle strip,
tetrahedron) pass `native_decide`.
-/

import Curvenet.Halfedge

namespace Curvenet
namespace HalfedgeBuilder

/-- Source vertex of interior halfedge `h`, given the triangle list. -/
def sourceOfInterior (tris : Array (Nat × Nat × Nat)) (h : Nat) : Nat :=
  let f := h / 3
  let l := h % 3
  let (t0, t1, t2) := tris[f]!
  if l = 0 then t0 else if l = 1 then t1 else t2

/-- Build a `HalfedgeMesh` from a vertex count and a flat triangle list.
   Each triangle yields 3 interior halfedges; orphan interior halfedges
   each get one paired boundary halfedge. Boundary halfedges are linked
   into their outer-face loop in step 4. -/
def fromTriangles (vertexCount : Nat) (tris : Array (Nat × Nat × Nat))
    : HalfedgeMesh := Id.run do
  let nTris := tris.size
  let nInterior := 3 * nTris
  let mut hes : Array Halfedge := Array.mkEmpty (nInterior * 2)
  -- Step 1: interior halfedges.
  for f in [0:nTris] do
    let (t0, t1, t2) := tris[f]!
    let base := 3 * f
    hes := hes.push ⟨t1, none, base + 1, some f⟩
    hes := hes.push ⟨t2, none, base + 2, some f⟩
    hes := hes.push ⟨t0, none, base, some f⟩
  -- Step 2: pair interior twins by (source, target).
  let mut needBoundary : Array Nat := #[]
  for i in [0:nInterior] do
    let tgt := hes[i]!.target
    let src := sourceOfInterior tris i
    -- Linear scan for halfedge j with target=src and source=tgt.
    let mut found : Option Nat := none
    for j in [0:nInterior] do
      if hes[j]!.target = src && sourceOfInterior tris j = tgt then
        found := some j
        break
    match found with
    | some j =>
        let h := hes[i]!
        hes := hes.set! i { h with twin := some j }
    | none =>
        needBoundary := needBoundary.push i
  -- Step 3: pair each interior orphan with a boundary halfedge.
  let boundaryStart := hes.size
  for kk in [0:needBoundary.size] do
    let i := needBoundary[kk]!
    let bhTarget := sourceOfInterior tris i
    let bhIdx := hes.size
    hes := hes.push ⟨bhTarget, some i, 0, none⟩
    let h := hes[i]!
    hes := hes.set! i { h with twin := some bhIdx }
  -- Step 4: link boundary halfedges into their outer-face loop.
  -- For boundary bh: source(bh) = he[twin(bh)].target. Then bh.next is the
  -- boundary halfedge whose source equals bh.target.
  let n := hes.size
  for i in [boundaryStart:n] do
    let myTarget := hes[i]!.target
    let mut nextIdx : Nat := i  -- self-loop fallback (open boundary endpoint)
    for j in [boundaryStart:n] do
      let bhJ := hes[j]!
      match bhJ.twin with
      | some twinJ =>
          let srcJ := hes[twinJ]!.target
          if srcJ = myTarget then
            nextIdx := j
            break
      | none => pure ()
    let bh' := hes[i]!
    hes := hes.set! i { bh' with next := nextIdx }
  return ⟨vertexCount, nTris, hes⟩

end HalfedgeBuilder

/- ============================================================ -/
/- Manifold + count proofs on tiny triangle inputs.            -/
/- ============================================================ -/

namespace HalfedgeBuilderExamples

open HalfedgeBuilder

/-- One triangle with verts (0, 1, 2). -/
def singleTri : HalfedgeMesh :=
  fromTriangles 3 #[(0, 1, 2)]

/-- Two triangles sharing edge (1, 2): (0,1,2) and (2,1,3). The shared
   edge has interior twins; the four other edges become boundary. -/
def twoTriStrip : HalfedgeMesh :=
  fromTriangles 4 #[(0, 1, 2), (2, 1, 3)]

/-- A tetrahedron — 4 verts, 4 triangles, no boundary halfedges. -/
def tet : HalfedgeMesh :=
  fromTriangles 4 #[(0, 1, 2), (0, 2, 3), (0, 3, 1), (1, 3, 2)]

/-- Single triangle: 3 verts, 1 face, 6 halfedges (3 interior + 3 boundary). -/
example : singleTri.vertexCount = 3 := by native_decide
example : singleTri.faceCount = 1 := by native_decide
example : singleTri.heCount = 6 := by native_decide
example : singleTri.manifold? = true := by native_decide

/-- Two-tri strip: 4 verts, 2 faces, 10 halfedges (6 interior + 4 boundary;
   the shared edge contributes interior twins, no boundary halfedges). -/
example : twoTriStrip.vertexCount = 4 := by native_decide
example : twoTriStrip.faceCount = 2 := by native_decide
example : twoTriStrip.heCount = 10 := by native_decide
example : twoTriStrip.manifold? = true := by native_decide

/-- Tetrahedron: 4 verts, 4 faces, 12 halfedges (all interior, no boundary). -/
example : tet.vertexCount = 4 := by native_decide
example : tet.faceCount = 4 := by native_decide
example : tet.heCount = 12 := by native_decide
example : tet.manifold? = true := by native_decide

/-- Twin involution on the tetrahedron: every halfedge has a twin and
   `twin (twin h) = h`. -/
example : tet.twinInvolutive = true := by native_decide

end HalfedgeBuilderExamples

end Curvenet
