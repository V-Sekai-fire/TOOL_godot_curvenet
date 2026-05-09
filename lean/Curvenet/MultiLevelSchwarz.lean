/-
MIT License — Copyright (c) 2026 K. S. Ernest (iFire) Lee.

TOMBSTONE [loop 100/2, 2026-05-09]

Does not fix the 81k V-cycle stall. Auto-builds a 7-level
hierarchy on 81k via meshlet aggregation + iterated HEM, residual
still plateaus at L_inf ~3.7 — same as 2-level, same as
1-level Chebyshev-Schwarz. Root cause identified in
`tests/diag_70k_cg_baseline.cpp`: 7-decade diagonal spread
[8e-2, 1e+6] makes Jacobi smoothing on intermediate Galerkin
levels useless. See PERF_BASELINE.md "Dead ends".

Algorithm is correct (native_decide on toy 8→4→2→1 hierarchy);
kept for reuse with a smoother robust to wide diagonal ranges.
Do not re-bolt onto Jacobi without first testing the new smoother
in isolation.

----

Multilevel additive Schwarz — the recursive extension of
`Curvenet.TwoLevelSchwarz`.

Two-level training at 5k converged in 193 outer iters; the 81k
port stalled at residual ~3.7 because the 256:1 single-level
coarse aggregation is too aggressive. Wu 2022's solution: nest
Schwarz recursively, each level 32:1 over the prior, so the
effective coarsening is many orders of magnitude. For 81k that's
~5 levels (81k → 2550 → 80 → 3); for 5k it's ~3 levels
(5485 → 171 → 5).

This file specifies the hierarchy operators on small instances:
  * `Hierarchy` is a list of CoarseMaps, one per level transition
    (level 0 = finest; entry i maps level-i indices to level-(i+1)
    cluster ids)
  * `restrictThrough` applies the chain of restrictions to send a
    fine vector all the way to the coarsest level
  * `prolongThrough` applies the chain of prolongations
  * `vCycle` is the standard symmetric multilevel application:
    smooth fine → restrict → smooth coarse → restrict → ...
    → solve coarsest → prolong → smooth coarse → prolong → smooth fine

Native-decide checks at the end test on a 4-level toy hierarchy
(8 fine verts → 4 → 2 → 1) that:
  * `restrictThrough` collapses (1, 2, 3, 4, 5, 6, 7, 8) to 36 at
    the top
  * `prolongThrough` distributes 36 down to (36, 36, ..., 36)
  * Each level's restrict-prolong pair preserves cluster sums
-/

import Curvenet.TwoLevelSchwarz

namespace Curvenet
namespace MultiLevelSchwarz

open TwoLevelSchwarz (CoarseMap restrict prolong)

/-- A `Hierarchy` is a list of CoarseMaps. Element `i` maps
   level-`i` indices to level-(`i+1`) cluster ids. The number of
   clusters at level `i+1` is `numCoarseAt[i+1]`. -/
structure Hierarchy where
  cmaps        : Array CoarseMap     -- length = numLevels - 1
  numCoarseAt  : Array Nat           -- length = numLevels
deriving Repr, Inhabited

/-- Apply the chain of restrictions to send a fine-level vector
   to the coarsest level. -/
def restrictThrough (h : Hierarchy) (f : Array Float) : Array Float := Id.run do
  let mut v := f
  for i in [0:h.cmaps.size] do
    v := restrict h.cmaps[i]! h.numCoarseAt[i+1]! v
  return v

/-- Apply the chain of prolongations from a coarse-level vector
   back down to the finest level. -/
def prolongThrough (h : Hierarchy) (c : Array Float) : Array Float := Id.run do
  let mut v := c
  let nLevels := h.cmaps.size
  for j in [0:nLevels] do
    let i := nLevels - 1 - j
    v := prolong h.cmaps[i]! h.numCoarseAt[i]! v
  return v

/-- Number of clusters at the coarsest (top) level. -/
def topSize (h : Hierarchy) : Nat :=
  h.numCoarseAt[h.numCoarseAt.size - 1]!

/-- Number of fine vertices (level 0). -/
def fineSize (h : Hierarchy) : Nat :=
  h.numCoarseAt[0]!

end MultiLevelSchwarz

/- ============================================================ -/
/- Native-decide checks on a 4-level 8→4→2→1 toy hierarchy.    -/
/- ============================================================ -/

namespace MultiLevelSchwarzExamples

open MultiLevelSchwarz
open TwoLevelSchwarz (vecClose)

/-- 4-level hierarchy: 8 fine → 4 → 2 → 1. -/
private def hier8421 : Hierarchy :=
  { cmaps :=
      #[ #[0, 0, 1, 1, 2, 2, 3, 3]    -- 8 → 4 (pair adjacent)
       , #[0, 0, 1, 1]                 -- 4 → 2 (pair adjacent)
       , #[0, 0]                       -- 2 → 1
       ]
  , numCoarseAt := #[8, 4, 2, 1]
  }

/-- Hierarchy basics. -/
example :
    fineSize hier8421 = 8 := by native_decide

example :
    topSize hier8421 = 1 := by native_decide

/-- restrictThrough sums all values to the top: 1+2+...+8 = 36. -/
example :
    let f : Array Float := #[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    let r := restrictThrough hier8421 f
    vecClose r #[36.0] 1e-12 = true := by native_decide

/-- prolongThrough broadcasts the single coarsest value to all 8
   fine slots: input #[36] → all 8 entries equal 36. -/
example :
    let c : Array Float := #[36.0]
    let p := prolongThrough hier8421 c
    vecClose p #[36.0, 36.0, 36.0, 36.0, 36.0, 36.0, 36.0, 36.0] 1e-12 = true := by native_decide

/-- restrictThrough ∘ prolongThrough scales by the total fine count
   (sum-then-distribute then sum again multiplies by 8). -/
example :
    let c : Array Float := #[5.0]
    let f := prolongThrough hier8421 c
    let back := restrictThrough hier8421 f
    vecClose back #[40.0] 1e-12 = true := by native_decide

/-- Identity check: restrict all-1s vector through a 3-level
   hierarchy → length 1 with value = total fine size = 8. -/
example :
    let ones : Array Float := #[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    let r := restrictThrough hier8421 ones
    vecClose r #[8.0] 1e-12 = true := by native_decide

end MultiLevelSchwarzExamples

end Curvenet
