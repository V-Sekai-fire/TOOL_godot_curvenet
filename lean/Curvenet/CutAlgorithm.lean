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
import Curvenet.CutMesh

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

/-- Split a face by connecting `target(a)` to `target(b)` along a new
   curvenet segment, where `a` and `b` are halfedges of the **same** face.

   Pre-conditions (caller's responsibility):
     * `m.halfedges[a].face = m.halfedges[b].face = some f`
     * `a ≠ b` and they are non-adjacent in the face loop (otherwise the
       split is degenerate — one of the resulting "faces" has zero area)
     * `target(a) ≠ target(b)`

   After split:
     * vertexCount unchanged, heCount + 2, faceCount + 1
     * Old face index `f` is reused for one sub-face (the loop containing
       h_b and the new BA halfedge).
     * New face index `nf` is assigned to the other sub-face (the loop
       containing h_a and the new AB halfedge).
     * Halfedges in the new sub-face are re-tagged from `some f` to
       `some nf`.
     * No cracks introduced, so the result still satisfies `manifold?`.
-/
def splitFace (m : HalfedgeMesh) (a b : Nat) : HalfedgeMesh := Id.run do
  let nh := m.heCount
  let nf := m.faceCount
  let heA := m.halfedges[a]!
  let heB := m.halfedges[b]!
  let oldNextA := heA.next
  let oldNextB := heB.next
  let f'  : Nat := nf
  let hAB : Nat := nh
  let hBA : Nat := nh + 1

  let mut newHalfedges := m.halfedges
  -- Re-tag halfedges from `oldNextB` up to (but not including) `a` so they
  -- belong to the new face f'.
  let mut cur := oldNextB
  let mut steps : Nat := 0
  while cur ≠ a ∧ steps < nh do
    let he := newHalfedges[cur]!
    newHalfedges := newHalfedges.set! cur { he with face := some f' }
    cur := he.next
    steps := steps + 1
  -- h_a joins the new face and points at the new AB halfedge.
  newHalfedges := newHalfedges.set! a { heA with face := some f', next := hAB }
  -- h_b stays in the old face but redirects to the new BA halfedge.
  newHalfedges := newHalfedges.set! b { heB with next := hBA }

  -- New AB halfedge: lives in face f', goes to target(b), then resumes the
  -- old face B loop via heB's original next.
  newHalfedges := newHalfedges.push
    { target := heB.target
    , twin   := some hBA
    , next   := oldNextB
    , face   := some f' }
  -- New BA halfedge: lives in face f, goes to target(a), then resumes the
  -- old face A loop via heA's original next.
  newHalfedges := newHalfedges.push
    { target := heA.target
    , twin   := some hAB
    , next   := oldNextA
    , face   := heA.face }

  return { vertexCount := m.vertexCount
         , faceCount   := nf + 1
         , halfedges   := newHalfedges }

/- CutMesh-level wrappers extend per-vertex kind and per-halfedge
   segment annotations alongside the underlying halfedge surgery.
   They live in the same `CutAlgorithm` namespace to avoid colliding
   with the `CutMesh` structure name. -/

/-- Wraps `subdivideEdge` while extending the cut-mesh annotations: the
   new vertex (index = old vertexCount) gets the kind `newVertexKind`
   (typically `sample` if a curvenet sample landed there or
   `edgeIntersection` for a generic chain crossing). The two new
   halfedges inherit the existing edge's segment annotation (`none` if
   it wasn't already on a curvenet segment). -/
def subdivideEdgeCM (cm : CutMesh) (h : Nat) (newVertexKind : CutVertexKind) : CutMesh :=
  let newBase := subdivideEdge cm.base h
  let newKinds := cm.vertexKind.push newVertexKind
  let segOfH := cm.segmentOfHalfedge[h]!
  let newSegs := cm.segmentOfHalfedge.push segOfH |>.push segOfH
  { base              := newBase
  , vertexKind        := newKinds
  , segmentOfHalfedge := newSegs }

/-- Wraps `splitFace`, tagging both new halfedges with the supplied
   curvenet segment id. Vertex kinds are unchanged; if the caller wants
   the connecting endpoints promoted to samples, that's a separate
   step. -/
def splitFaceCM (cm : CutMesh) (a b : Nat) (segId : Nat) : CutMesh :=
  let newBase := splitFace cm.base a b
  let newSegs := cm.segmentOfHalfedge.push (some segId) |>.push (some segId)
  { base              := newBase
  , vertexKind        := cm.vertexKind
  , segmentOfHalfedge := newSegs }

/-- Insert a "crack" — a dead-end slit branching from a face-boundary
   vertex into the face interior. `h_anchor` is a halfedge in the face
   whose target is the boundary vertex the crack branches from. The new
   vertex (index = old vertexCount) sits at the slit's far end inside
   the face, with no other edges incident to it.

   Adds +1 vertex, +2 halfedges; the face count is unchanged. The two
   new halfedges are twins of each other and BOTH point to the same
   face (the §4.1 crack configuration). The face's halfedge loop is
   extended to traverse the slit (out, then back); loop length grows
   by 2.

   Pre-conditions (caller's responsibility):
     * `m.halfedges[h_anchor].face = some f` (the crack lives inside f).

   The standard `manifold?` invariants survive the operation because:
     * twin involution holds for the new (out, in) pair
     * source(out) = target(in) = v_boundary (slit endpoints meet at
       the boundary vertex)
     * the face loop is still closed; just longer
-/
def insertCrack (m : HalfedgeMesh) (h_anchor : Nat) : HalfedgeMesh := Id.run do
  let nh := m.heCount
  let nv := m.vertexCount
  let heAnchor := m.halfedges[h_anchor]!
  let vBoundary := heAnchor.target
  let oldNext := heAnchor.next
  let cIdx : Nat := nv
  let hOut : Nat := nh       -- v_boundary → c (going into the slit)
  let hIn  : Nat := nh + 1   -- c → v_boundary (returning from the slit)

  let mut newHalfedges := m.halfedges
  -- h_anchor now redirects into the slit instead of the next boundary halfedge.
  newHalfedges := newHalfedges.set! h_anchor { heAnchor with next := hOut }
  -- hOut: leaves the boundary vertex toward the slit endpoint.
  newHalfedges := newHalfedges.push
    { target := cIdx
    , twin   := some hIn
    , next   := hIn
    , face   := heAnchor.face }
  -- hIn: returns from the slit endpoint to the boundary vertex, then
  -- resumes the original face loop.
  newHalfedges := newHalfedges.push
    { target := vBoundary
    , twin   := some hOut
    , next   := oldNext
    , face   := heAnchor.face }

  return { vertexCount := nv + 1
         , faceCount   := m.faceCount
         , halfedges   := newHalfedges }

/-- CutMesh wrapper for `insertCrack`: extends vertex kind (the new
   isolated vertex is typically a curvenet sample, hence the
   `newVertexKind` argument) and tags both crack halfedges with the
   supplied segment id. -/
def insertCrackCM (cm : CutMesh) (h_anchor : Nat) (segId : Nat)
    (newVertexKind : CutVertexKind) : CutMesh :=
  let newBase := insertCrack cm.base h_anchor
  let newKinds := cm.vertexKind.push newVertexKind
  let newSegs  :=
    cm.segmentOfHalfedge.push (some segId) |>.push (some segId)
  { base              := newBase
  , vertexKind        := newKinds
  , segmentOfHalfedge := newSegs }

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

/- ============================================================ -/
/- Face split: bisect the unit quad along the (1, 3) diagonal  -/
/- by connecting halfedge 0 (target=1) to halfedge 2 (target=3).-/
/- Result: two triangle faces, no cracks, still manifold.       -/
/- ============================================================ -/

def quadDiagonalSplit : HalfedgeMesh := splitFace Examples.quad 0 2

example : quadDiagonalSplit.vertexCount = 4 := by native_decide
example : quadDiagonalSplit.heCount     = 10 := by native_decide
example : quadDiagonalSplit.faceCount   = 2 := by native_decide
example : quadDiagonalSplit.manifold?   = true := by native_decide

/-- Each of the two new sub-faces is a triangle (3-halfedge loop). -/
example :
    let m := quadDiagonalSplit
    -- Face A loop starting at halfedge 1 (which stayed in face 0).
    let lenA := Id.run do
      let mut cur := m.halfedges[1]!.next
      let mut steps : Nat := 1
      while cur ≠ 1 ∧ steps < m.heCount do
        cur := m.halfedges[cur]!.next
        steps := steps + 1
      return steps
    -- Face B loop starting at halfedge 0 (now in the new face 1).
    let lenB := Id.run do
      let mut cur := m.halfedges[0]!.next
      let mut steps : Nat := 1
      while cur ≠ 0 ∧ steps < m.heCount do
        cur := m.halfedges[cur]!.next
        steps := steps + 1
      return steps
    (lenA = 3 ∧ lenB = 3) := by native_decide

/-- Euler characteristic V − E + F = 2 also holds after a face split
   (we add 0 V, 1 E, 1 F so the alternating sum is unchanged). -/
example :
    Examples.quad.vertexCount + (Examples.quad.faceCount + 1)
        - (Examples.quad.heCount / 2) =
    quadDiagonalSplit.vertexCount + (quadDiagonalSplit.faceCount + 1)
        - (quadDiagonalSplit.heCount / 2) := by native_decide

/-- Composition: subdivide an edge, then split the resulting face.
   Builds an N=5 polygon split into a triangle + a quad. -/
def quadSubdivThenSplit : HalfedgeMesh :=
  let m := subdivideEdge Examples.quad 0   -- 5 verts, 10 he, 1 face
  splitFace m 1 3                            -- split via diagonal of resulting pentagon

example : quadSubdivThenSplit.vertexCount = 5 := by native_decide
example : quadSubdivThenSplit.heCount     = 12 := by native_decide
example : quadSubdivThenSplit.faceCount   = 2 := by native_decide
example : quadSubdivThenSplit.manifold?   = true := by native_decide

/- ============================================================ -/
/- CutMesh wrappers: surgery + annotation extension preserves   -/
/- partition-of-unity and VtCIsZero.                            -/
/- ============================================================ -/

private def noSampleCol : Nat → Nat → Bool → Nat := fun _ _ _ => 0

/-- Subdivide the uncut quad's edge 0 with the new vertex tagged as a
   curvenet sample. The CutMesh now has 5 vertices (one promoted) and
   10 halfedges. -/
def quadSubdivCM : CutMesh :=
  CutAlgorithm.subdivideEdgeCM CutExamples.uncutQuad 0
    (CutVertexKind.sample 0 0 false)

example : quadSubdivCM.vertexCount = 5 := by native_decide
example : quadSubdivCM.heCount     = 10 := by native_decide

/-- Partition of unity holds across the subdivided cut-mesh: every
   halfedge contributes to exactly one of V or C. -/
example : quadSubdivCM.partitionOfUnity noSampleCol = true := by native_decide

/-- VtC = 0 also still holds. -/
example : quadSubdivCM.VtCIsZero noSampleCol = true := by native_decide

/-- Splitting the uncut quad's diagonal as a CutMesh keeps the
   annotations consistent: the two new halfedges are tagged with
   segment 7, vertex kinds unchanged, partition of unity intact. -/
def quadSplitCM : CutMesh :=
  CutAlgorithm.splitFaceCM CutExamples.uncutQuad 0 2 7

example : quadSplitCM.heCount = 10 := by native_decide

/-- Both new halfedges have segmentOfHalfedge = some 7. -/
example :
    let cm := quadSplitCM
    (cm.segmentOfHalfedge[8]! == some 7 ∧
     cm.segmentOfHalfedge[9]! == some 7) := by native_decide

example : quadSplitCM.partitionOfUnity noSampleCol = true := by native_decide
example : quadSplitCM.VtCIsZero        noSampleCol = true := by native_decide

/- ============================================================ -/
/- Crack insertion: dead-end slit inside a face anchored at a   -/
/- face-boundary vertex. The standard manifold? invariants still -/
/- hold because the new pair forms a valid twin and source/      -/
/- target match at the boundary vertex.                          -/
/- ============================================================ -/

/-- Insert a crack into the unit triangle anchored at halfedge 0
   (target = vertex 1). The slit's far end is the new vertex 3. -/
def triangleWithCrack : HalfedgeMesh := insertCrack Examples.triangle 0

example : triangleWithCrack.vertexCount = 4 := by native_decide
example : triangleWithCrack.heCount     = 8 := by native_decide
example : triangleWithCrack.faceCount   = 1 := by native_decide

/-- Crack insertion preserves the standard manifold invariants. -/
example : triangleWithCrack.manifold? = true := by native_decide

/-- Both crack halfedges point into the same face — the §4.1 crack
   configuration. Halfedge indices 6 (out) and 7 (in) both have
   `face = some 0`, and they are each other's twins. -/
example :
    let m  := triangleWithCrack
    let h6 := m.halfedges[6]!
    let h7 := m.halfedges[7]!
    h6.face == some 0 && h7.face == some 0 &&
    h6.twin == some 7 && h7.twin == some 6 := by
  native_decide

/-- Face 0's loop now has 5 halfedges (was 3) — the slit traversal
   adds 2. Walk via `.next`. -/
example :
    let m := triangleWithCrack
    let len := Id.run do
      let mut cur := m.halfedges[0]!.next
      let mut steps : Nat := 1
      while cur ≠ 0 ∧ steps < m.heCount do
        cur := m.halfedges[cur]!.next
        steps := steps + 1
      return steps
    len = 5 := by native_decide

/-- CutMesh wrapper: inserting a crack tagged with segment id 11 and
   marking the new vertex as a sample. -/
def quadWithCrackCM : CutMesh :=
  CutAlgorithm.insertCrackCM CutExamples.uncutQuad 0 11
    (CutVertexKind.sample 0 0 false)

example : quadWithCrackCM.vertexCount = 5 := by native_decide
example : quadWithCrackCM.heCount     = 10 := by native_decide
example : quadWithCrackCM.partitionOfUnity noSampleCol = true := by native_decide
example : quadWithCrackCM.VtCIsZero        noSampleCol = true := by native_decide

/- ============================================================ -/
/- Segment chain across multiple faces (Fig. 6 col 4): the      -/
/- segment crosses one mesh edge between two faces. Decomposes  -/
/- into one subdivideEdge + two splitFace calls on the two     -/
/- adjacent faces.                                              -/
/- ============================================================ -/

/-- Two-triangle strip with a chain-1 segment from vertex 0 to vertex 3,
   crossing the shared 1-2 edge. The segment decomposes as:

     1. subdivideEdge on the shared edge (halfedge 1) — adds vertex 4
     2. splitFace face 0 between he 2 (target = vertex 0) and he 1
        (target = vertex 4 after subdivision) — adds the in-face
        segment from start to the crossing
     3. splitFace face 1 between he 3 (target = vertex 4 after
        subdivision) and he 4 (target = vertex 3) — adds the in-face
        segment from the crossing to the end -/
def twoTriStripWithChain : HalfedgeMesh :=
  let m1 := CutAlgorithm.subdivideEdge Examples.twoTriStrip 1
  let m2 := CutAlgorithm.splitFace m1 2 1
  CutAlgorithm.splitFace m2 3 4

example : twoTriStripWithChain.vertexCount = 5 := by native_decide

/-- Halfedge count: original 10 + 2 (edge subdivision) + 2 (face 0 split)
   + 2 (face 1 split) = 16. -/
example : twoTriStripWithChain.heCount = 16 := by native_decide

/-- Face count: original 2 + 1 (face 0 split) + 1 (face 1 split) = 4. -/
example : twoTriStripWithChain.faceCount = 4 := by native_decide

/-- Manifold invariants survive the entire chain. -/
example : twoTriStripWithChain.manifold? = true := by native_decide

/-- Each of the four resulting sub-faces is a triangle (3-halfedge loop).
   Walk face 0's loop starting at halfedge 0; expect length 3. -/
example :
    let m := twoTriStripWithChain
    let len := Id.run do
      let mut cur := m.halfedges[0]!.next
      let mut steps : Nat := 1
      while cur ≠ 0 ∧ steps < m.heCount do
        cur := m.halfedges[cur]!.next
        steps := steps + 1
      return steps
    len = 3 := by native_decide

end CutAlgorithmExamples

end Curvenet
