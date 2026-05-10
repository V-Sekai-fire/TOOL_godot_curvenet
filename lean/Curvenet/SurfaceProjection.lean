/-
MIT License — Copyright (c) 2026 K. S. Ernest (iFire) Lee.

Curvenet-knot to mesh-vertex projection. Mirrors
`src/curvenet/surface_projection.h`.

The C++ side currently only implements VERTEX projection (closest mesh
vertex by squared distance). Edge and face projections are TODO (slices
10/12 and 11/20 respectively). This Lean spec mirrors that scope: we
verify the closest-vertex search and the column-assignment pass that
collapses multiple knots projecting to the same vertex onto a single
sample-column.

Edge / face projection kinds are kept in the type for forward
compatibility but never produced by `projectToVertices`.
-/

import Curvenet.Vec3

namespace Curvenet
namespace SurfaceProjection

inductive ProjectionKind
  | vertex
  | edgeIntersection
  | faceInterior
deriving Repr, Inhabited, BEq, DecidableEq

structure ProjectedKnot where
  kind      : ProjectionKind
  meshIndex : Int        -- vertex_idx for kind = vertex; -1 if no match
  position  : Vec3
deriving Repr, Inhabited

/-- Squared distance between two points. Local because importing the
   helper from `CurvenetBuilder` would pull in the whole graph type. -/
def distSq (a b : Vec3) : Float :=
  let dx := a.x - b.x
  let dy := a.y - b.y
  let dz := a.z - b.z
  dx * dx + dy * dy + dz * dz

/-- Project each input knot to its closest mesh vertex. Returns one
   `ProjectedKnot` per input knot, all of kind `vertex`. If the mesh is
   empty, `meshIndex = -1` and `position` is the input knot. -/
def projectToVertices (knots : Array Vec3) (meshPositions : Array Vec3)
    : Array ProjectedKnot := Id.run do
  let mut out : Array ProjectedKnot := Array.mkEmpty knots.size
  for i in [0:knots.size] do
    let k := knots[i]!
    let mut bestIdx : Int := -1
    let mut bestD : Float := 1e308  -- effectively +∞
    for v in [0:meshPositions.size] do
      let d := distSq meshPositions[v]! k
      if d < bestD then
        bestD := d
        bestIdx := Int.ofNat v
    let pos := if bestIdx ≥ 0 then meshPositions[bestIdx.toNat]! else k
    out := out.push ⟨ProjectionKind.vertex, bestIdx, pos⟩
  return out

/-- Column assignment pass: each unique projected vertex gets a sequential
   column index; knots projecting to the same vertex share a column. -/
def promoteVertexSamples (vertexCount : Nat) (projections : Array ProjectedKnot)
    : Array Int := Id.run do
  let mut vertexToCol : Array Int := Array.replicate vertexCount (-1)
  let mut knotToCol : Array Int := Array.replicate projections.size (-1)
  let mut nextCol : Int := 0
  for i in [0:projections.size] do
    let pk := projections[i]!
    if pk.kind == ProjectionKind.vertex && pk.meshIndex ≥ 0 then
      let v := pk.meshIndex.toNat
      let cur := vertexToCol[v]!
      if cur < 0 then
        vertexToCol := vertexToCol.set! v nextCol
        knotToCol := knotToCol.set! i nextCol
        nextCol := nextCol + 1
      else
        knotToCol := knotToCol.set! i cur
  return knotToCol

end SurfaceProjection

/- ============================================================ -/
/- Concrete cases.                                              -/
/- ============================================================ -/

namespace SurfaceProjectionExamples

open SurfaceProjection

private def origin : Vec3 := ⟨0.0, 0.0, 0.0⟩
private def xUnit  : Vec3 := ⟨1.0, 0.0, 0.0⟩
private def yUnit  : Vec3 := ⟨0.0, 1.0, 0.0⟩

private def triMesh : Array Vec3 := #[origin, xUnit, yUnit]

/-- A knot exactly at vertex 0 projects to vertex 0. -/
example :
    let r := projectToVertices #[origin] triMesh
    r.size = 1 ∧ r[0]!.meshIndex = 0 := by native_decide

/-- A knot near (0.9, 0.0, 0.0) is closer to xUnit (vertex 1) than to
   origin or yUnit. -/
example :
    let r := projectToVertices #[⟨0.9, 0.0, 0.0⟩] triMesh
    r[0]!.meshIndex = 1 := by native_decide

/-- A knot near (0.0, 0.9, 0.0) projects to yUnit (vertex 2). -/
example :
    let r := projectToVertices #[⟨0.0, 0.9, 0.0⟩] triMesh
    r[0]!.meshIndex = 2 := by native_decide

/-- All output kinds are `vertex` for vertex-only projection. -/
example :
    let r := projectToVertices #[origin, xUnit, yUnit] triMesh
    r.size = 3 ∧
    r[0]!.kind = ProjectionKind.vertex ∧
    r[1]!.kind = ProjectionKind.vertex ∧
    r[2]!.kind = ProjectionKind.vertex := by native_decide

/-- Empty knot list yields empty output. -/
example :
    let r := projectToVertices #[] triMesh
    r.size = 0 := by native_decide

/-- Empty mesh: meshIndex stays `-1`, position falls back to the knot. -/
example :
    let r := projectToVertices #[origin] #[]
    r[0]!.meshIndex = -1 := by native_decide

/-- Two knots projecting to the same vertex share a column. -/
example :
    let projections := projectToVertices #[origin, ⟨0.01, 0.0, 0.0⟩] triMesh
    let cols := promoteVertexSamples 3 projections
    -- Both knots are closest to origin (vertex 0), so both get column 0.
    cols = #[0, 0] := by native_decide

/-- Knots projecting to distinct vertices get distinct columns in encounter
   order. -/
example :
    let projections := projectToVertices #[xUnit, yUnit, origin] triMesh
    let cols := promoteVertexSamples 3 projections
    cols = #[0, 1, 2] := by native_decide

/-- Knot order matters: encountering vertex 2 first still assigns col 0 to
   it, then col 1 to vertex 0. -/
example :
    let projections := projectToVertices #[yUnit, origin] triMesh
    let cols := promoteVertexSamples 3 projections
    cols = #[0, 1] := by native_decide

/-- Repeated knot at the same vertex yields a repeated column. -/
example :
    let projections := projectToVertices #[origin, xUnit, origin] triMesh
    let cols := promoteVertexSamples 3 projections
    cols = #[0, 1, 0] := by native_decide

end SurfaceProjectionExamples

end Curvenet
