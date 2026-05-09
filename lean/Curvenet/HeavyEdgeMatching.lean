/-
MIT License — Copyright (c) 2026 K. S. Ernest (iFire) Lee.

TOMBSTONE [loop 100/1, 2026-05-09]

Does not fix the 81k V-cycle stall. Connectivity-aware HEM
aggregation produced the same L_inf residual plateau ~3.7 as
principal-axis bucketing did. Aggregation quality is not the
bottleneck; the 7-decade diagonal spread is (see
`tests/diag_70k_cg_baseline.cpp` and PERF_BASELINE.md
"Dead ends").

Algorithm is correct (native_decide on path/star/cycle/path-8);
kept as drop-in coarsener for any future multilevel try with a
better smoother.

----

Heavy-edge matching aggregation — connectivity-aware coarsening for
Multi-level Schwarz.

Loop-8 diagnosis: principal-axis bucketing of meshlet centroids on
a T-pose body slices through head, both arms, and torso
simultaneously, producing near-disconnected Galerkin blocks. The
multilevel V-cycle stalls at residual ~3.73 on the 81k mesh
because the coarse correction can't propagate information across
disconnected aggregates.

Heavy-edge matching (HEM, Karypis-Kumar 1998) fixes this: at every
node, pair with an adjacent unmatched neighbor. Aggregates are
*always* connected by construction, so the Galerkin coarse matrix
is well-conditioned. One pass yields 2:1 coarsening; iterating the
pass gives 2^k:1 coarsening for any depth.

This file specifies the algorithm on tiny graphs:

  Adjacency : list of neighbor lists, one per node.
  hemMatch   : Adjacency -> Array Nat (cmap from node id to cluster
              id; node i and node j share a cluster iff they were
              matched in this pass).

  Properties verified by native_decide:
    * cmap length = node count
    * empty graph: every node is its own singleton
    * path 0-1-2-3: pairs become (0, 1) and (2, 3) -> 2 clusters
    * star (0 -- 1, 0 -- 2, 0 -- 3): 0 pairs with 1, leaves 2 & 3
      become singletons -> 3 clusters total
    * cycle 0-1-2-3-0: pairs become (0, 1) and (2, 3) -> 2 clusters
    * iterated HEM: applying twice on path 0..7 yields 2 clusters
-/

namespace Curvenet
namespace HeavyEdgeMatching

abbrev Adjacency := Array (Array Nat)

/-- One pass of heavy-edge matching: walk nodes in index order, pair
   each unmatched node with the first unmatched neighbor, otherwise
   leave it as a singleton. Returns a cmap of length `adj.size` whose
   values are dense (every cluster id < numCoarse). -/
def hemMatch (adj : Adjacency) : Array Nat × Nat := Id.run do
  let n := adj.size
  let mut cmap : Array Nat := Array.replicate n 0
  let mut matched : Array Bool := Array.replicate n false
  let mut nextId : Nat := 0
  for i in [0:n] do
    if matched[i]! then continue
    let nbrs := adj[i]!
    let mut paired := false
    for j in nbrs do
      if j = i then continue
      if matched[j]! then continue
      cmap := cmap.set! i nextId
      cmap := cmap.set! j nextId
      matched := matched.set! i true
      matched := matched.set! j true
      nextId := nextId + 1
      paired := true
      break
    if !paired then
      cmap := cmap.set! i nextId
      matched := matched.set! i true
      nextId := nextId + 1
  return (cmap, nextId)

/-- Iterate `hemMatch` `k` times to get up to 2^k:1 coarsening.
   The output is a *single* cmap from the original fine indices to
   the final cluster ids. Each iteration coarsens the previous adj
   graph by composing per-cluster neighbor sets. -/
def iterateHem (adj0 : Adjacency) (k : Nat) : Array Nat × Nat := Id.run do
  -- We track the cumulative cmap from original indices to current
  -- cluster ids, plus the current coarsened adjacency.
  let n0 := adj0.size
  let mut cmap : Array Nat := (Array.range n0)   -- identity initially
  let mut adj : Adjacency := adj0
  let mut numC := n0
  for _ in [0:k] do
    let (step, n_next) := hemMatch adj
    -- Compose: new[i] := step[old[i]]
    cmap := cmap.map (fun c => step[c]!)
    -- Build coarsened adjacency: cluster c is adjacent to cluster d
    -- iff some node in c is adjacent to some node in d (and c != d).
    let mut nbrSets : Array (Array Nat) := Array.replicate n_next #[]
    for i in [0:adj.size] do
      let ci := step[i]!
      for j in adj[i]! do
        let cj := step[j]!
        if ci ≠ cj then
          let curr := nbrSets[ci]!
          if !curr.contains cj then
            nbrSets := nbrSets.set! ci (curr.push cj)
    adj := nbrSets
    numC := n_next
  return (cmap, numC)

end HeavyEdgeMatching

/- ============================================================ -/
/- Native-decide checks on small graphs.                        -/
/- ============================================================ -/

namespace HeavyEdgeMatchingExamples

open HeavyEdgeMatching

/-- Empty 4-node graph (no edges): every node becomes a singleton. -/
private def emptyAdj : Adjacency := #[#[], #[], #[], #[]]

example : (hemMatch emptyAdj).1.size = 4 := by native_decide
example : (hemMatch emptyAdj).2 = 4 := by native_decide
example : (hemMatch emptyAdj).1 = #[0, 1, 2, 3] := by native_decide

/-- Path 0 -- 1 -- 2 -- 3: walking left to right, 0 pairs with 1
   (its first neighbor), then 2 pairs with 3. Two clusters. -/
private def pathAdj : Adjacency :=
  #[ #[1], #[0, 2], #[1, 3], #[2] ]

example : (hemMatch pathAdj).2 = 2 := by native_decide
example : (hemMatch pathAdj).1 = #[0, 0, 1, 1] := by native_decide

/-- Star: 0 is the center connected to 1, 2, 3. 0 pairs with 1
   (first neighbor); 2 and 3 are unmatched (their only neighbor 0
   is already matched), so they become singletons. -/
private def starAdj : Adjacency :=
  #[ #[1, 2, 3], #[0], #[0], #[0] ]

example : (hemMatch starAdj).2 = 3 := by native_decide
example : (hemMatch starAdj).1 = #[0, 0, 1, 2] := by native_decide

/-- 4-cycle 0 -- 1 -- 2 -- 3 -- 0. Same outcome as the path: 0 pairs
   with 1 (first neighbor), then 2 pairs with 3. -/
private def cycleAdj : Adjacency :=
  #[ #[1, 3], #[0, 2], #[1, 3], #[0, 2] ]

example : (hemMatch cycleAdj).2 = 2 := by native_decide
example : (hemMatch cycleAdj).1 = #[0, 0, 1, 1] := by native_decide

/-- Iterated HEM on path 0..7 (8 nodes). One pass: (0,1)(2,3)(4,5)(6,7)
   -> 4 clusters. Two passes coarsen the 4-cluster path graph: after
   the first pass adjacencies are 0-1-2-3 (cluster path), so the
   second pass produces 2 clusters (0,1)(2,3). Final cmap on the
   original 8 nodes: #[0,0,0,0,1,1,1,1]. -/
private def path8Adj : Adjacency :=
  #[ #[1], #[0, 2], #[1, 3], #[2, 4], #[3, 5], #[4, 6], #[5, 7], #[6] ]

example : (iterateHem path8Adj 1).2 = 4 := by native_decide
example : (iterateHem path8Adj 1).1 = #[0, 0, 1, 1, 2, 2, 3, 3] := by native_decide
example : (iterateHem path8Adj 2).2 = 2 := by native_decide
example : (iterateHem path8Adj 2).1 = #[0, 0, 0, 0, 1, 1, 1, 1] := by native_decide

end HeavyEdgeMatchingExamples

end Curvenet
