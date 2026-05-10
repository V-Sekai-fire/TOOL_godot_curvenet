/-
MIT License — Copyright (c) 2026 K. S. Ernest (iFire) Lee.

Hierarchical Sparsify and Compensate (HSC) — Krishnan, Fattal,
Szeliski 2013, `KrishnanFattalSzeliski2013HSC` in references.bib.

Loops 8 → 100/9 ruled out the entire iterative-PCG family at our
target scale (proven in docs/IMPOSSIBILITY.md): plain CG, ICC(0),
multilevel additive Schwarz, HEM aggregation, Chebyshev,
block-CG, lockstep multi-RHS — none reduces κ(M⁻¹A) below the
~480 ceiling required to fit 5 ms / 12-RHS at 50k verts.

HSC is the algorithmically right answer for graph Laplacians
because it adds the Schur-compensation step the multilevel
Schwarz family lacked. Each level eliminates a chosen subset of
vertices and "compensates" by adjusting the remaining edge
weights so that the reduced Laplacian is the Schur complement
of the original — preserving the matrix's action on the
remaining DOFs while shrinking the system.

This file specifies the load-bearing per-vertex elimination
primitive on a tiny weighted graph:

  Graph as adjacency list of (neighbor, weight) pairs.
  eliminateVertex (g : Graph) (v : Nat) : Graph

  When vertex v is eliminated:
    1. For each pair (i, j) of v's neighbors:
       new edge weight w_new(i, j) = w_old(i, j) + (w(v, i) · w(v, j)) / sum_k w(v, k)
    2. Vertex v is removed; all incident edges are removed.

  This is the standard Gaussian-elimination of vertex v from the
  graph Laplacian L_G — the resulting Laplacian on V \ {v} is
  exactly the Schur complement L_G[V', V'] minus the L[V', v]
  L[v, v]^{-1} L[v, V'] correction.

Native-decide checks at the end verify on a 3-vertex path graph
(0—1—2) and a 4-vertex star (0 center, 1, 2, 3 leaves) that
elimination produces the correct compensated weights.
-/

namespace Curvenet
namespace HSC

/-- An undirected weighted graph as a flat edge list. Each edge
   `(u, v, w)` with u < v and weight w > 0. Self-loops disallowed.
   This is the simplest representation that's easy to reason about
   in `native_decide` examples. The runtime C++ mirror uses CSR
   for performance. -/
structure Graph where
  numVerts : Nat
  edges    : Array (Nat × Nat × Float)
deriving Repr, Inhabited

/-- Sum of weights of all edges incident to vertex v. -/
def degreeWeight (g : Graph) (v : Nat) : Float := Id.run do
  let mut s : Float := 0.0
  for e in g.edges do
    let (u, w, ew) := e
    if u = v ∨ w = v then
      s := s + ew
  return s

/-- The list of (neighbor, weight) pairs for vertex v. -/
def neighborsOf (g : Graph) (v : Nat) : Array (Nat × Float) := Id.run do
  let mut acc : Array (Nat × Float) := #[]
  for e in g.edges do
    let (u, w, ew) := e
    if u = v then
      acc := acc.push (w, ew)
    else if w = v then
      acc := acc.push (u, ew)
  return acc

/-- Find the weight of edge (a, b), or 0 if absent. -/
def edgeWeight (g : Graph) (a b : Nat) : Float := Id.run do
  let lo := if a < b then a else b
  let hi := if a < b then b else a
  for e in g.edges do
    let (u, w, ew) := e
    if u = lo ∧ w = hi then
      return ew
  return 0.0

/-- Eliminate vertex v from graph g, producing the Schur-
   complement-equivalent graph on the remaining vertices. The
   compensation rule per Krishnan-Fattal-Szeliski 2013 §3.1:
   for each pair of v's neighbors (i, j), the edge weight
   between them is updated by
     w_new(i, j) = w_old(i, j) + w(v, i) · w(v, j) / deg(v)
   (where deg(v) is the sum of v's incident weights).
   Edges incident to v are removed from the output. -/
def eliminateVertex (g : Graph) (v : Nat) : Graph := Id.run do
  let nbrs := neighborsOf g v
  let dv := degreeWeight g v
  -- Start by copying every edge that doesn't touch v.
  let mut newEdges : Array (Nat × Nat × Float) := #[]
  for e in g.edges do
    let (u, w, ew) := e
    if u ≠ v ∧ w ≠ v then
      newEdges := newEdges.push (u, w, ew)
  -- Apply Schur compensation on every pair of v's neighbors.
  if dv > 0.0 then
    for i in [0:nbrs.size] do
      for j in [i+1:nbrs.size] do
        let (ni, wi) := nbrs[i]!
        let (nj, wj) := nbrs[j]!
        let lo := if ni < nj then ni else nj
        let hi := if ni < nj then nj else ni
        let comp := wi * wj / dv
        -- Find existing edge in newEdges and accumulate, or push.
        let mut found := false
        for k in [0:newEdges.size] do
          let (a, b, ek) := newEdges[k]!
          if a = lo ∧ b = hi then
            newEdges := newEdges.set! k (a, b, ek + comp)
            found := true
            break
        if !found then
          newEdges := newEdges.push (lo, hi, comp)
  return { numVerts := g.numVerts, edges := newEdges }

/-- Tolerance-aware Float comparison. -/
def fclose (x y eps : Float) : Bool := (x - y).abs < eps

end HSC

/- ============================================================ -/
/- Native-decide checks on tiny weighted graphs.                 -/
/- ============================================================ -/

namespace HSCExamples

open HSC

/-- Path graph 0 ─(1.0)─ 1 ─(2.0)─ 2.
   Eliminating vertex 1 with neighbors {0 (w=1.0), 2 (w=2.0)}:
     deg(1) = 3.0
     compensated edge (0, 2): 0 + (1·2)/3 = 0.6666...
   Result: a single edge (0, 2, 0.6666...). -/
private def path3 : Graph :=
  { numVerts := 3,
    edges    := #[ (0, 1, 1.0), (1, 2, 2.0) ] }

example :
    let g' := eliminateVertex path3 1
    g'.edges.size = 1 := by native_decide

example :
    let g' := eliminateVertex path3 1
    let w := edgeWeight g' 0 2
    fclose w (2.0 / 3.0) 1e-12 = true := by native_decide

/-- Star graph 0 (center) connected to 1, 2, 3 with weights 1, 2, 3.
   Eliminating vertex 0 with neighbors {1 (w=1), 2 (w=2), 3 (w=3)}:
     deg(0) = 6.0
     compensated edges:
       (1, 2): (1·2)/6 = 0.333...
       (1, 3): (1·3)/6 = 0.5
       (2, 3): (2·3)/6 = 1.0
   Result: triangle on {1, 2, 3}. -/
private def star4 : Graph :=
  { numVerts := 4,
    edges    := #[ (0, 1, 1.0), (0, 2, 2.0), (0, 3, 3.0) ] }

example :
    let g' := eliminateVertex star4 0
    g'.edges.size = 3 := by native_decide

example :
    let g' := eliminateVertex star4 0
    fclose (edgeWeight g' 1 2) (1.0 / 3.0) 1e-12 = true := by native_decide

example :
    let g' := eliminateVertex star4 0
    fclose (edgeWeight g' 1 3) 0.5 1e-12 = true := by native_decide

example :
    let g' := eliminateVertex star4 0
    fclose (edgeWeight g' 2 3) 1.0 1e-12 = true := by native_decide

/-- Triangle 0-1-2 (all weights 1.0) with an extra leaf 3 attached
   to 0 (weight 4.0). Eliminating 0:
     neighbors of 0: {1 (w=1), 2 (w=1), 3 (w=4)}, deg = 6
     existing edge (1, 2) with weight 1.0
     compensated (1, 2): 1 + (1·1)/6 = 1.1666...
     compensated (1, 3): 0 + (1·4)/6 = 0.6666...
     compensated (2, 3): 0 + (1·4)/6 = 0.6666...
   Verify the existing edge accumulates with the compensation. -/
private def triPlusLeaf : Graph :=
  { numVerts := 4,
    edges    := #[ (0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0),
                   (0, 3, 4.0) ] }

example :
    let g' := eliminateVertex triPlusLeaf 0
    fclose (edgeWeight g' 1 2) (1.0 + 1.0 / 6.0) 1e-12 = true := by native_decide

example :
    let g' := eliminateVertex triPlusLeaf 0
    fclose (edgeWeight g' 1 3) (4.0 / 6.0) 1e-12 = true := by native_decide

end HSCExamples

end Curvenet
