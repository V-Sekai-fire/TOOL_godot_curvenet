# Cycle 10 #4 — README refresh

**File:** `README.md`

## Current state

- Cycle 4 row reads: `N-gon patch (N=3 + N=4; N≥5 TODO)` — still says TODO.
- `ngon_patch.h` description: `N!=4 TODO mean-value`.
- Property total: `48 properties × 100 random cases = 4,800 checks passing` — outdated. Live count is **55** (verified via `make -C tests test 2>&1 | grep -c "OK, passed"`).
- Lean total: `17 instance theorems` — needs to grow by **7** (added in commit `7d9bb2a`: 5 corner-Lagrange + partition-of-unity + symmetric-centroid). Plus whatever todo #2 adds.

## Goal

Update the status table and totals after todos #1–#3 are green:

- Cycle 4 row → green, drop the `N≥5 TODO` qualifier.
- Refresh property count to current total.
- Refresh Lean theorem count.
- Update the `src/curvenet/` architecture comment for `ngon_patch.h` so it stops claiming N≥5 is unimplemented.
