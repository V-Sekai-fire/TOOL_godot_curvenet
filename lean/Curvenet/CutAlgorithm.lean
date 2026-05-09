/-
MIT License — Copyright (c) 2026 K. S. Ernest (iFire) Lee.

Cut-mesh surgery primitives — slice 10 of the DeGoes22 rewrite.

DeGoes22 §4.1 enumerates four arrangements between a curvenet segment
and the underlying mesh polygon (Figure 6). This slice implements the
simplest one: an **edge subdivision** when a curvenet sample lies on a
mesh edge.

  * `subdivideEdge m h` inserts a new vertex on the edge containing
    halfedge `h`, splitting both `h` and its twin into two halfedges
    each. The result has +1 vertex and +2 halfedges; the manifold
    invariants of slice 1 are preserved.

Future slices will add:
  * face split  — segment lying inside a single mesh face (Fig. 6 col 2)
  * face cracks — segment isolated inside a face (Fig. 6 col 3)
  * segment chain — segment crossing multiple polygons (Fig. 6 col 4)

Edge subdivision is the only case that doesn't introduce a crack, so the
result still satisfies `manifold?`. Slices for the crack-introducing
cases will validate against a relaxed `cutMeshWellFormed?` predicate
since cracks intentionally break `twinsAreOpposite`.
-/

import Curvenet.Halfedge

namespace Curvenet
namespace CutAlgorithm

/-- Insert a new cut-vertex on the edge containing halfedge `h`,
   splitting `h` into `h ; h_new` and its twin into `twin ; twin_new`.

   Pre-conditions (not Lean-checked here): `h` is in range, the edge has
   a twin (no boundary edges with twin = none in our model — boundary
   halfedges still have twins pointing to the virtual outer face).

   Numbering convention:
     * new vertex index   = old vertexCount
     * new halfedge `h_new`     = old heCount     (the second half on h's side)
     * new halfedge `twin_new`  = old heCount + 1 (the second half on twin side)

   Twin pairing after subdivision:
     h            ↔ twin_new
     h_new        ↔ twin
-/
def subdivideEdge (m : HalfedgeMesh) (h : Nat) : HalfedgeMesh := Id.run do
  let nh := m.heCount
  let nv := m.vertexCount
  let heOrig := m.halfedges[h]!
  let twinIdx : Nat := match heOrig.twin with
    | some t => t
    | none   => h  -- caller bug; degenerate but keep things total
  let heTwin := m.halfedges[twinIdx]!
  let cIdx       : Nat := nv
  let hNewIdx    : Nat := nh
  let twinNewIdx : Nat := nh + 1

  -- h     : src(h) -> c
  let hOrigUpdated : Halfedge :=
    { target := cIdx
    , twin   := some twinNewIdx
    , next   := hNewIdx
    , face   := heOrig.face }
  -- h_new : c -> old target of h
  let hNewBuilt : Halfedge :=
    { target := heOrig.target
    , twin   := some twinIdx
    , next   := heOrig.next
    , face   := heOrig.face }
  -- twin  : src(twin) -> c
  let hTwinUpdated : Halfedge :=
    { target := cIdx
    , twin   := some hNewIdx
    , next   := twinNewIdx
    , face   := heTwin.face }
  -- twin_new : c -> old target of twin (i.e., src(h))
  let hTwinNewBuilt : Halfedge :=
    { target := heTwin.target
    , twin   := some h
    , next   := heTwin.next
    , face   := heTwin.face }

  let mut newHalfedges := m.halfedges
  newHalfedges := newHalfedges.set! h hOrigUpdated
  newHalfedges := newHalfedges.set! twinIdx hTwinUpdated
  newHalfedges := newHalfedges.push hNewBuilt
  newHalfedges := newHalfedges.push hTwinNewBuilt

  return { vertexCount := nv + 1
         , faceCount   := m.faceCount
         , halfedges   := newHalfedges }

end CutAlgorithm

/- ============================================================ -/
/- Subdividing edge 0 of the triangle yields a 4-vertex,        -/
/- 8-halfedge manifold mesh with the affected face loop length  -/
/- bumped from 3 to 4.                                           -/
/- ============================================================ -/

namespace CutAlgorithmExamples

open CutAlgorithm

/-- Triangle with edge 0 (the 0→1 halfedge) subdivided. -/
def triangleSplit : HalfedgeMesh := subdivideEdge Examples.triangle 0

example : triangleSplit.vertexCount = 4 := by native_decide
example : triangleSplit.heCount     = 8 := by native_decide
example : triangleSplit.faceCount   = 1 := by native_decide

/-- Subdivision preserves manifold invariants (no cracks introduced). -/
example : triangleSplit.manifold? = true := by native_decide

/-- The interior face loop now has 4 halfedges (was 3). Walk via `next`. -/
example :
    let m := triangleSplit
    -- Start at he 0 (still in face 0), count steps until we return.
    let len := Id.run do
      let mut cur := m.halfedges[0]!.next
      let mut steps : Nat := 1
      while cur ≠ 0 ∧ steps < m.heCount do
        cur := m.halfedges[cur]!.next
        steps := steps + 1
      return steps
    len = 4 := by native_decide

/-- Subdividing the unit quad's edge 0 gives 5 vertices, 10 halfedges,
   still manifold. -/
def quadSplit : HalfedgeMesh := subdivideEdge Examples.quad 0

example : quadSplit.vertexCount = 5 := by native_decide
example : quadSplit.heCount     = 10 := by native_decide
example : quadSplit.manifold?   = true := by native_decide

/-- Subdividing twice on different edges still yields a manifold mesh. -/
def triangleSplitTwice : HalfedgeMesh :=
  subdivideEdge (subdivideEdge Examples.triangle 0) 1

example : triangleSplitTwice.vertexCount = 5 := by native_decide
example : triangleSplitTwice.heCount     = 10 := by native_decide
example : triangleSplitTwice.manifold?   = true := by native_decide

/-- Euler characteristic V − E + F = 2 stays invariant under edge
   subdivision (we add 1 V, 1 E, 0 F). -/
example :
    Examples.triangle.vertexCount + (Examples.triangle.faceCount + 1)
        - (Examples.triangle.heCount / 2) =
    triangleSplit.vertexCount + (triangleSplit.faceCount + 1)
        - (triangleSplit.heCount / 2) := by native_decide

end CutAlgorithmExamples

end Curvenet
