/-
MIT License — Copyright (c) 2026 K. S. Ernest (iFire) Lee.

Halfedge mesh data structure — slice 1 of the DeGoes22 "Character Articulation
through Profile Curves" rewrite. Subsequent slices add the cut-mesh (which
permits halfedge cracks) and the polygonal Laplacian assembly.

Manifold invariant. The DeGoes22 algorithm (§4.1) requires an oriented
manifold input mesh, possibly with boundary. We expose this as a Boolean
checker `manifold?` so concrete instance meshes can be verified via
`native_decide`. At runtime the C++ side will satisfy the invariant by
running fTetWild (or an equivalent manifold-repair pass) against the input
mesh — see `todos/06_ftetwild_runtime_integration.md`.

Conventions matching the paper:
  * `Halfedge.target` is the vertex this halfedge points TO (head).
  * `Halfedge.next`   is the next halfedge inside the same face loop, CCW.
  * `Halfedge.twin`   is the opposite halfedge across the underlying edge,
                      `none` if this halfedge sits on a mesh boundary.
  * `Halfedge.face`   is the index of the face this halfedge bounds, `none`
                      for boundary halfedges (a single virtual outer face).

We do NOT yet model cracks (twin pointing inside the same face); those are
introduced in `Curvenet/CutMesh.lean` once we begin tracing curvenet
segments through the surface.
-/

namespace Curvenet

structure Halfedge where
  target : Nat
  twin   : Option Nat
  next   : Nat
  face   : Option Nat
deriving Repr, Inhabited, BEq

/-- Halfedge mesh. `vertexCount` is the explicit upper bound on `target`
   indices (we don't need vertex *positions* for the structural invariants). -/
structure HalfedgeMesh where
  vertexCount : Nat
  faceCount   : Nat
  halfedges   : Array Halfedge
deriving Repr, Inhabited

namespace HalfedgeMesh

@[inline] def heCount (m : HalfedgeMesh) : Nat := m.halfedges.size

/-- Each halfedge index is in-range. -/
def indicesInRange (m : HalfedgeMesh) : Bool := Id.run do
  let n := m.heCount
  for i in [0:n] do
    let h := m.halfedges[i]!
    if h.target ≥ m.vertexCount then return false
    if h.next ≥ n then return false
    match h.twin with
    | some t => if t ≥ n then return false
    | none => pure ()
    match h.face with
    | some f => if f ≥ m.faceCount then return false
    | none => pure ()
  return true

/-- `twin` is an involution where defined: `twin (twin h) = some h`. -/
def twinInvolutive (m : HalfedgeMesh) : Bool := Id.run do
  for i in [0:m.heCount] do
    let h := m.halfedges[i]!
    match h.twin with
    | none => pure ()
    | some t =>
        let ht := m.halfedges[t]!
        match ht.twin with
        | some i' => if i' ≠ i then return false
        | none => return false
  return true

/-- Twins point to opposite ends: `m.halfedges[twin h].target` is the source
   of `h`, i.e. `m.halfedges[twin (prev h)].target = h.target` is wrong;
   instead the cleanest invariant is that the source vertex of a halfedge
   equals the target of its twin. The source of `h` is the target of the
   halfedge whose `next` is `h`. We check this by walking face loops and
   matching twin endpoints. -/
def twinsAreOpposite (m : HalfedgeMesh) : Bool := Id.run do
  let n := m.heCount
  -- Build prev[h] := the halfedge whose .next equals h. In a well-formed
  -- mesh this is unique per halfedge.
  let mut prev : Array Nat := Array.replicate n 0
  for i in [0:n] do
    let nxt := m.halfedges[i]!.next
    prev := prev.set! nxt i
  for i in [0:n] do
    let h := m.halfedges[i]!
    match h.twin with
    | none => pure ()
    | some t =>
        let src := m.halfedges[prev[i]!]!.target
        let tgt := m.halfedges[t]!.target
        if src ≠ tgt then return false
  return true

/-- Following `.next` from any halfedge of an interior face returns to the
   starting halfedge in at most `heCount` steps, and every step belongs to
   the same face. -/
def faceLoopsClose (m : HalfedgeMesh) : Bool := Id.run do
  let n := m.heCount
  for start in [0:n] do
    let h0 := m.halfedges[start]!
    match h0.face with
    | none => pure ()  -- boundary halfedges form their own outer loop; skip
    | some f0 =>
        let mut cur := h0.next
        let mut steps : Nat := 0
        let mut closed : Bool := false
        while steps < n do
          if cur = start then
            closed := true
            break
          if m.halfedges[cur]!.face ≠ some f0 then
            return false
          cur := m.halfedges[cur]!.next
          steps := steps + 1
        if !closed then return false
  return true

/-- A halfedge mesh is manifold iff every structural check holds. Cracks
   (which violate `twinsAreOpposite`) are NOT permitted at this layer. -/
def manifold? (m : HalfedgeMesh) : Bool :=
  m.indicesInRange && m.twinInvolutive && m.twinsAreOpposite && m.faceLoopsClose

end HalfedgeMesh

/- ============================================================ -/
/- Concrete tiny meshes. Each is verified manifold via         -/
/- `native_decide`.                                            -/
/- ============================================================ -/

namespace Examples

/-- A single triangle (3 verts, 1 interior face, 3 boundary halfedges).
   Halfedges 0,1,2 are interior (face = some 0); 3,4,5 are boundary
   (face = none) and form the outer loop in reverse. -/
def triangle : HalfedgeMesh :=
  { vertexCount := 3
  , faceCount   := 1
  , halfedges   := #[
      -- interior CCW loop 0->1, 1->2, 2->0
      ⟨1, some 5, 1, some 0⟩,  -- 0: 0->1
      ⟨2, some 4, 2, some 0⟩,  -- 1: 1->2
      ⟨0, some 3, 0, some 0⟩,  -- 2: 2->0
      -- boundary CW loop 0->2, 2->1, 1->0 (twin of 2,1,0 in that order)
      ⟨2, some 2, 4, none⟩,    -- 3: 0->2
      ⟨1, some 1, 5, none⟩,    -- 4: 2->1
      ⟨0, some 0, 3, none⟩     -- 5: 1->0
    ]
  }

/-- A single quad (4 verts, 1 interior face). -/
def quad : HalfedgeMesh :=
  { vertexCount := 4
  , faceCount   := 1
  , halfedges   := #[
      ⟨1, some 7, 1, some 0⟩,  -- 0: 0->1
      ⟨2, some 6, 2, some 0⟩,  -- 1: 1->2
      ⟨3, some 5, 3, some 0⟩,  -- 2: 2->3
      ⟨0, some 4, 0, some 0⟩,  -- 3: 3->0
      ⟨3, some 3, 5, none⟩,    -- 4: 0->3
      ⟨2, some 2, 6, none⟩,    -- 5: 3->2
      ⟨1, some 1, 7, none⟩,    -- 6: 2->1
      ⟨0, some 0, 4, none⟩     -- 7: 1->0
    ]
  }

/-- Two triangles sharing edge 1-2 (4 verts, 2 interior faces). The shared
   edge has twinned interior halfedges; the four other edges are boundary. -/
def twoTriStrip : HalfedgeMesh :=
  { vertexCount := 4
  , faceCount   := 2
  , halfedges   := #[
      -- Face 0: 0->1->2->0
      ⟨1, some 7, 1, some 0⟩,  -- 0: 0->1
      ⟨2, some 3, 2, some 0⟩,  -- 1: 1->2  (interior twin with halfedge 3)
      ⟨0, some 6, 0, some 0⟩,  -- 2: 2->0
      -- Face 1: 2->1->3->2
      ⟨1, some 1, 4, some 1⟩,  -- 3: 2->1  (interior twin with halfedge 1)
      ⟨3, some 9, 5, some 1⟩,  -- 4: 1->3
      ⟨2, some 8, 3, some 1⟩,  -- 5: 3->2
      -- Boundary loop CCW around the outer face: 0->2 -> 2->3 -> 3->1 -> 1->0.
      ⟨2, some 2, 8, none⟩,    -- 6: 0->2
      ⟨0, some 0, 6, none⟩,    -- 7: 1->0
      ⟨3, some 5, 9, none⟩,    -- 8: 2->3
      ⟨1, some 4, 7, none⟩     -- 9: 3->1
    ]
  }

end Examples

/- The triangle, quad, and two-triangle strip all pass `manifold?`. -/

example : Examples.triangle.manifold?    = true := by native_decide
example : Examples.quad.manifold?        = true := by native_decide
example : Examples.twoTriStrip.manifold? = true := by native_decide

/-- Sanity check on Halfedge counts: a single triangle has 6 halfedges
   (3 interior + 3 boundary). -/
example : Examples.triangle.heCount = 6 := by native_decide

/-- A single quad has 8 halfedges (4 interior + 4 boundary). -/
example : Examples.quad.heCount = 8 := by native_decide

/-- Euler-characteristic-ish: V − E + F = 2 for a topological disk
   counting the outer face. For a single triangle V=3, E=3, F=2. -/
example : Examples.triangle.vertexCount + (Examples.triangle.faceCount + 1) - (Examples.triangle.heCount / 2) = 2 := by native_decide

/-- Same for the quad: V=4, E=4, F=2 ⇒ 4−4+2 = 2. -/
example : Examples.quad.vertexCount + (Examples.quad.faceCount + 1) - (Examples.quad.heCount / 2) = 2 := by native_decide

end Curvenet
