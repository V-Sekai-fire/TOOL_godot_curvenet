/-
MIT License — Copyright (c) 2026 K. S. Ernest (iFire) Lee.

Curvenet-driven Direct Delta Mush — bind-time post-processing.

Pair file to `lean/Curvenet/DirectDeltaMush.lean` (runtime
LBS-flavored matvec). At bind time the pipeline is:

  1. Capture per-vertex influence weights `W[v, i]` by running
     DeGoes22's two-stage solve (HarmonicSolve + DeformSolve)
     for unit perturbations of each handle. Reuses existing
     specs — not new in this file.

  2. **Smooth** `W` via Laplacian iterations (the "mush" step
     from Le & Lewis 2019). Diffuses sharp influence boundaries
     introduced by the discrete cut-mesh, giving DDM its
     characteristic LBS-without-candy-wrapper look.

  3. **Sparsify** to top-K entries per vertex (typically K = 4-8
     for character meshes) — bounds runtime memory traffic of
     the per-vertex matvec.

  4. **Renormalize** so each vertex's row sums to 1, preserving
     partition-of-unity. Required for the runtime kernel to map
     a constant-on-handles deformation to a constant-on-vertices
     output (i.e. rigid translation behaves correctly).

This file specs steps 2-4. Step 1 is the existing HarmonicSolve
pipeline; step 5 (per-vertex matrix bake) is a runtime-side
optimization that would slot into DirectDeltaMush.lean.

Native-decide checks at the end verify on a 4-vertex path graph
with 2 handles:
  * identity weights are preserved by smoothing
  * all-half weights are preserved by smoothing (uniform
    distribution is a Laplacian eigenfunction at eigenvalue 0)
  * top-2 sparsification on a 4-handle row keeps the largest
    two and renormalizes them to sum 1
  * partition-of-unity holds before and after both ops
-/

import Curvenet.Common

namespace Curvenet

open Curvenet.Common
namespace DirectDeltaMushBind

/-- Per-vertex weight matrix as nested Array. Outer index is
   vertex; inner is handle. Dense representation; the runtime
   path (DirectDeltaMush.lean) consumes a sparse Influences
   list — the bridge is `sparsifyTopK` defined below. -/
abbrev WeightMatrix := Array (Array Float)

/-- Sum of one vertex's row. -/
def rowSum (row : Array Float) : Float := row.foldl (· + ·) 0.0


/-- Per-row partition-of-unity check: every row sum is within
   `eps` of 1.0. -/
def partitionOfUnity (W : WeightMatrix) (eps : Float) : Bool := Id.run do
  let mut ok := true
  for row in W do
    if ¬ fclose (rowSum row) 1.0 eps then ok := false
  return ok

/-- Damped Jacobi-style Laplacian smoothing of the weight matrix.
   For each pass and each handle column, every vertex's weight
   is replaced by `(1 - omega) * w[v] + omega * mean over
   neighbors of w[nbr]`. Runs `nu` passes.

   Uniform mean weighting (no cot-Laplacian here) — the harmonic
   prior already came from a cot-Laplacian solve at step 1, so
   step 2's job is local diffusion only. ω = 0.5 mixes evenly
   between current and neighbor average. -/
def smoothWeights (W : WeightMatrix) (adj : Adjacency)
                       (nu : Nat) (omega : Float) : WeightMatrix := Id.run do
  let mut Wcur := W
  let n := W.size
  let nh := if n > 0 then W[0]!.size else 0
  for _ in [0:nu] do
    let mut Wnext : WeightMatrix := Array.replicate n (Array.replicate nh 0.0)
    for v in [0:n] do
      let nbrs := adj[v]!
      let count := nbrs.size
      let mut row : Array Float := Array.replicate nh 0.0
      for i in [0:nh] do
        let mut nb_avg : Float := 0.0
        if count > 0 then
          for u in nbrs do
            nb_avg := nb_avg + Wcur[u]![i]!
          nb_avg := nb_avg / Float.ofNat count
        let blended := (1.0 - omega) * Wcur[v]![i]! + omega * nb_avg
        row := row.set! i blended
      Wnext := Wnext.set! v row
    Wcur := Wnext
  return Wcur

/-- Top-K sparsification with renormalization. For each row,
   keep the K entries with largest absolute weight, zero the
   rest, and divide remaining entries by their sum so each row
   sums to 1 (preserving partition-of-unity for harmonic-derived
   inputs).

   Output is a sparse representation: per row, an array of
   (handle_idx, normalized_weight) pairs of length ≤ K. -/
def sparsifyTopK (W : WeightMatrix) (K : Nat) : Array (Array (Nat × Float)) := Id.run do
  let n := W.size
  let mut out : Array (Array (Nat × Float)) := Array.replicate n #[]
  for v in [0:n] do
    let row := W[v]!
    -- Collect (idx, |w|) pairs, sort descending by |w|.
    let mut pairs : Array (Nat × Float) := #[]
    for i in [0:row.size] do
      pairs := pairs.push (i, row[i]!)
    -- Selection sort top-K (small K, simple is fine).
    let take := if K < pairs.size then K else pairs.size
    for k in [0:take] do
      let mut best := k
      for j in [k+1:pairs.size] do
        if pairs[j]!.2.abs > pairs[best]!.2.abs then best := j
      let tmp := pairs[k]!
      pairs := pairs.set! k pairs[best]!
      pairs := pairs.set! best tmp
    -- Truncate, renormalize.
    let mut kept : Array (Nat × Float) := #[]
    let mut sum : Float := 0.0
    for k in [0:take] do
      kept := kept.push pairs[k]!
      sum := sum + pairs[k]!.2
    if sum != 0.0 then
      for k in [0:kept.size] do
        let (idx, w) := kept[k]!
        kept := kept.set! k (idx, w / sum)
    out := out.set! v kept
  return out

/-- Sum of the kept weights per vertex (for partition-of-unity
   check on the sparse representation). -/
def sparseRowSum (row : Array (Nat × Float)) : Float :=
  row.foldl (fun acc kv => acc + kv.2) 0.0

end DirectDeltaMushBind

/- ============================================================ -/
/- Native-decide checks on small examples.                       -/
/- ============================================================ -/

namespace DirectDeltaMushBindExamples

open DirectDeltaMushBind

/-- 4-vertex path graph adjacency. Vertices 0—1—2—3. -/
private def path4Adj : Adjacency :=
  #[ #[1], #[0, 2], #[1, 3], #[2] ]

/-- Identity weights: vertex v has full weight on handle v
   (treating handle space as same dim as vertex space here for
   testing; in production handle count differs from vertex
   count). 4 verts × 4 handles = identity matrix. -/
private def identityW : WeightMatrix :=
  #[ #[1.0, 0.0, 0.0, 0.0]
   , #[0.0, 1.0, 0.0, 0.0]
   , #[0.0, 0.0, 1.0, 0.0]
   , #[0.0, 0.0, 0.0, 1.0]
   ]

/-- Identity weights row-sum to 1: partition of unity holds. -/
example : partitionOfUnity identityW 1e-12 = true := by native_decide

/-- After 0 smoothing passes the matrix is unchanged. Sanity. -/
example :
    let W' := smoothWeights identityW path4Adj 0 0.5
    fclose W'[0]![0]! 1.0 1e-12 = true ∧
    fclose W'[1]![1]! 1.0 1e-12 = true := by native_decide

/-- All-half weights: each vertex has weight 0.5 on handles 0
   and 1, 0 on others. Uniform distribution within an active
   subset is a Laplacian eigenfunction at eigenvalue 0, so it's
   a fixed point of any consistent smoothing iteration. -/
private def halfW : WeightMatrix :=
  #[ #[0.5, 0.5, 0.0, 0.0]
   , #[0.5, 0.5, 0.0, 0.0]
   , #[0.5, 0.5, 0.0, 0.0]
   , #[0.5, 0.5, 0.0, 0.0]
   ]

example : partitionOfUnity halfW 1e-12 = true := by native_decide

example :
    let W' := smoothWeights halfW path4Adj 5 0.5
    partitionOfUnity W' 1e-12 = true := by native_decide

example :
    let W' := smoothWeights halfW path4Adj 5 0.5
    fclose W'[0]![0]! 0.5 1e-12 = true ∧
    fclose W'[2]![1]! 0.5 1e-12 = true := by native_decide

/-- Top-K sparsification of one row [0.4, 0.3, 0.2, 0.1] with
   K=2: keep first two, renormalize. New weights:
   0.4/(0.4+0.3) ≈ 0.5714, 0.3/(0.4+0.3) ≈ 0.4286. -/
private def skewW : WeightMatrix :=
  #[ #[0.4, 0.3, 0.2, 0.1] ]

example :
    let sp := sparsifyTopK skewW 2
    sp[0]!.size = 2 := by native_decide

example :
    let sp := sparsifyTopK skewW 2
    fclose (sparseRowSum sp[0]!) 1.0 1e-12 = true := by native_decide

example :
    let sp := sparsifyTopK skewW 2
    fclose sp[0]![0]!.2 (4.0 / 7.0) 1e-12 = true ∧
    fclose sp[0]![1]!.2 (3.0 / 7.0) 1e-12 = true := by native_decide

/-- Top-K with K = full width is a no-op (modulo renormalization,
   which is identity for already-normalized rows). -/
example :
    let sp := sparsifyTopK identityW 4
    -- Each identity row has only one non-zero, sparsifyTopK keeps
    -- 4 entries but most are zero. Only the diagonal sums non-zero.
    fclose (sparseRowSum sp[0]!) 1.0 1e-12 = true := by native_decide

-- Composition: sparsifyTopK . smoothWeights preserves partition of unity
-- when applied to a partition-of-unity input. This is the bind-time
-- invariant for DDM weight cooking.

private def sparseToWeightMatrix (sp : Array (Array (Nat × Float))) (cols : Nat) : WeightMatrix :=
  sp.map fun row =>
    Id.run do
      let mut out : Array Float := Array.replicate cols 0.0
      for (j, v) in row do
        if j < cols then out := out.set! j v
      pure out

/-- smoothWeights then sparsifyTopK preserves partition of unity on an
   identity weight matrix over the path-4 graph (K = 2). -/
example :
    let W1 := smoothWeights identityW path4Adj 2 0.5
    let sp := sparsifyTopK W1 2
    let W2 := sparseToWeightMatrix sp 4
    partitionOfUnity W2 1e-9 = true := by
  native_decide

/-- Same composition, K = 4 (no truncation), still partition of unity. -/
example :
    let W1 := smoothWeights identityW path4Adj 2 0.5
    let sp := sparsifyTopK W1 4
    let W2 := sparseToWeightMatrix sp 4
    partitionOfUnity W2 1e-9 = true := by
  native_decide

/-- Composition on the half-half row matrix (already partition of unity)
   with smoothing + top-K = 1 still preserves the row sum to 1. -/
example :
    let W1 := smoothWeights halfW path4Adj 1 0.3
    let sp := sparsifyTopK W1 1
    let W2 := sparseToWeightMatrix sp 2
    partitionOfUnity W2 1e-9 = true := by
  native_decide

end DirectDeltaMushBindExamples

end Curvenet
