/-
MIT License — Copyright (c) 2026 K. S. Ernest (iFire) Lee.

Cut-mesh data layer — slice 3 of the DeGoes22 rewrite.

Models the result of cutting a halfedge mesh by curvenet segments
(§4.1 of DeGoes22). Function values are stored per halfedge (face-corner)
so a single cut-vertex can hold *different* values from its incident
halfedges — that's how curve-side discontinuities are encoded.

Each cut-vertex carries a kind tag:
  * meshVertex          — coincident with an original mesh vertex
  * sample              — a curvenet sample point with a side-tag
  * edgeIntersection    — produced by a segment crossing a mesh edge

Each halfedge optionally carries a curvenet segment annotation; halfedges
that lie on a traced curvenet segment are the ones that pick up sample
values from the C operator.

Two sparse linear maps live on top of the halfedge-corner space:
  V : nh × nv  copies a per-mesh-vertex unknown to every incident halfedge
  C : nh × nc  copies a per-sample-side constraint to every incident halfedge

The DeGoes22 invariants we check here:
  V · 1_v + C · 1_c = 1_h        (partition of unity over halfedges)
  Vᵀ C = 0                        (curvenet samples take precedence over
                                   mesh-vertex unknowns)

Cracks (twins inside the same face) are permitted at the cut-mesh layer
but not exercised by the tests in this slice — those land with the
cutting algorithm in a future slice.
-/

import Curvenet.Halfedge

namespace Curvenet

/-- What a cut-vertex represents in the cut-mesh. -/
inductive CutVertexKind
  | meshVertex
  | sample          (curveId : Nat) (sampleIdx : Nat) (side : Bool)
  | edgeIntersection (mesh_edge : Nat)
deriving Repr, Inhabited, BEq

structure CutMesh where
  base                 : HalfedgeMesh
  /-- Indexed by base.vertexCount; what each cut-vertex represents. -/
  vertexKind           : Array CutVertexKind
  /-- Indexed by base.heCount; `some seg_id` if the halfedge lies on a
     traced curvenet segment, else `none`. -/
  segmentOfHalfedge    : Array (Option Nat)
deriving Repr, Inhabited

namespace CutMesh

@[inline] def heCount     (m : CutMesh) : Nat := m.base.heCount
@[inline] def vertexCount (m : CutMesh) : Nat := m.base.vertexCount

/-- For halfedge `h`, the column of V where the value 1 lives, or `none` if
   the halfedge maps to a curvenet constraint instead. Convention: V copies
   from the *target* mesh-vertex of each halfedge (the face-corner the
   halfedge points to). Halfedges whose target is a curvenet sample yield
   `none` so that constraint flows through C, not V. -/
def vColumnOf (m : CutMesh) (h : Nat) : Option Nat :=
  let he := m.base.halfedges[h]!
  match m.vertexKind[he.target]! with
  | CutVertexKind.meshVertex => some he.target
  | _                        => none

/-- For halfedge `h`, the column of C (per-sample-side) where the value 1
   lives, or `none` if no curvenet constraint applies. The column index is
   computed by the supplied `sampleColumn` map, which packs (curveId,
   sampleIdx, side) into a unique nc-range column. -/
def cColumnOf (m : CutMesh) (h : Nat)
    (sampleColumn : Nat → Nat → Bool → Nat) : Option Nat :=
  let he := m.base.halfedges[h]!
  match m.vertexKind[he.target]! with
  | CutVertexKind.sample c s side => some (sampleColumn c s side)
  | _                              => none

/-- Partition of unity: for every halfedge, exactly one of V or C contributes
   a 1, and never both. -/
def partitionOfUnity (m : CutMesh)
    (sampleColumn : Nat → Nat → Bool → Nat) : Bool := Id.run do
  for h in [0:m.heCount] do
    let v := m.vColumnOf h
    let c := m.cColumnOf h sampleColumn
    -- Exactly one of (some, none) configurations, never (some, some) or (none, none).
    match v, c with
    | some _, none => pure ()
    | none, some _ => pure ()
    | _,    _      => return false
  return true

/-- Vᵀ C = 0 — no halfedge contributes to both V and C. Trivially implied by
   `partitionOfUnity` but we expose it as an explicit named check. -/
def VtCIsZero (m : CutMesh)
    (sampleColumn : Nat → Nat → Bool → Nat) : Bool := Id.run do
  for h in [0:m.heCount] do
    let v := m.vColumnOf h
    let c := m.cColumnOf h sampleColumn
    if v.isSome && c.isSome then return false
  return true

end CutMesh

/- ============================================================ -/
/- Concrete uncut meshes lift the slice 1 examples cleanly:    -/
/- every cut-vertex is a mesh-vertex, no halfedge is on any    -/
/- segment. C is empty, V is a permutation of identity, and    -/
/- partition-of-unity / VᵀC = 0 are immediate.                 -/
/- ============================================================ -/

namespace CutExamples

open Examples

/-- Uncut triangle: 3 mesh-vertices, no segments. -/
def uncutTriangle : CutMesh :=
  { base              := triangle
  , vertexKind        := #[CutVertexKind.meshVertex, CutVertexKind.meshVertex, CutVertexKind.meshVertex]
  , segmentOfHalfedge := #[none, none, none, none, none, none]
  }

/-- Uncut quad: 4 mesh-vertices, no segments. -/
def uncutQuad : CutMesh :=
  { base              := quad
  , vertexKind        := #[CutVertexKind.meshVertex, CutVertexKind.meshVertex,
                            CutVertexKind.meshVertex, CutVertexKind.meshVertex]
  , segmentOfHalfedge := #[none, none, none, none, none, none, none, none]
  }

/-- Two-triangle strip, uncut. -/
def uncutTwoTriStrip : CutMesh :=
  { base              := twoTriStrip
  , vertexKind        := #[CutVertexKind.meshVertex, CutVertexKind.meshVertex,
                            CutVertexKind.meshVertex, CutVertexKind.meshVertex]
  , segmentOfHalfedge := Array.replicate 10 none
  }

/-- A trivial sample-column packer (uncut meshes never call it but the
   API needs one). -/
private def noSamples : Nat → Nat → Bool → Nat := fun _ _ _ => 0

/-- For an uncut mesh, every halfedge has a V column equal to its target. -/
example :
    let m := uncutTriangle
    (Array.ofFn (n := m.heCount) (fun (h : Fin m.heCount) =>
      m.vColumnOf h.val == some (m.base.halfedges[h.val]!.target))).all id = true := by native_decide

example :
    let m := uncutQuad
    (Array.ofFn (n := m.heCount) (fun (h : Fin m.heCount) =>
      m.vColumnOf h.val == some (m.base.halfedges[h.val]!.target))).all id = true := by native_decide

/-- Partition of unity on uncut meshes. -/
example : uncutTriangle.partitionOfUnity     CutExamples.noSamples = true := by native_decide
example : uncutQuad.partitionOfUnity         CutExamples.noSamples = true := by native_decide
example : uncutTwoTriStrip.partitionOfUnity  CutExamples.noSamples = true := by native_decide

/-- VᵀC = 0 on uncut meshes. -/
example : uncutTriangle.VtCIsZero     CutExamples.noSamples = true := by native_decide
example : uncutQuad.VtCIsZero         CutExamples.noSamples = true := by native_decide
example : uncutTwoTriStrip.VtCIsZero  CutExamples.noSamples = true := by native_decide

/- ============================================================ -/
/- Hand-constructed cut: triangle with a single curvenet sample -/
/- replacing one of its mesh vertices. Verifies that V loses    -/
/- one column and C picks up one, and the partition-of-unity    -/
/- still holds.                                                 -/
/- ============================================================ -/

/-- Triangle where vertex 0 has been promoted to a curvenet sample
   (curveId=0, sampleIdx=0, left side). All halfedges with target=0 now
   route through C; the others stay in V. -/
def triangleWithSample : CutMesh :=
  { base              := triangle
  , vertexKind        := #[CutVertexKind.sample 0 0 false,
                            CutVertexKind.meshVertex,
                            CutVertexKind.meshVertex]
  , segmentOfHalfedge := #[none, none, none, none, none, none]
  }

/-- Trivial sample packer: maps the single sample (0, 0, _) to column 0. -/
private def oneSample : Nat → Nat → Bool → Nat := fun _ _ _ => 0

example : triangleWithSample.partitionOfUnity CutExamples.oneSample = true := by native_decide
example : triangleWithSample.VtCIsZero        CutExamples.oneSample = true := by native_decide

/-- Halfedges with target=0 (vertex 0 was promoted to a sample) now have
   `vColumnOf = none` and `cColumnOf = some 0`. -/
example :
    let m := triangleWithSample
    -- halfedges with target 0: he 2 (2→0) interior, he 5 (1→0) boundary.
    m.vColumnOf 2 == none && m.cColumnOf 2 CutExamples.oneSample == some 0 = true := by native_decide

example :
    let m := triangleWithSample
    m.vColumnOf 5 == none && m.cColumnOf 5 CutExamples.oneSample == some 0 = true := by native_decide

end CutExamples

end Curvenet
