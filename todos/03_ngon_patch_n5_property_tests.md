# Cycle 10 #3 — RapidCheck properties for N≥5 patch

**File:** `tests/test_ngon_patch.cpp` (extend existing)

## Current state

`tests/test_ngon_patch.cpp` covers N=3 and N=4 only. Pure mean-value math is covered separately in `tests/test_mean_value.cpp` (4 properties, green).

## Goal

Add property tests for the N≥5 path of `NgonPatch::evaluate`, gated on the implementation from todo #1.

## Candidate properties

1. **Corner recovery** — for random `n ∈ [5, 8]` and a regular-N-gon-shaped boundary loop with linear cubics, evaluating at each vertex parameter returns `boundaries[k].c0` (within tight float eps).
2. **Boundary recovery** — for a query parameter on side k of the regular N-gon, the output equals `boundaries[k].evaluate(s_k)` (within eps).
3. **Translation invariance** — translating every boundary curve by `d` translates `NgonPatch::evaluate` by `d` for any interior query.
4. **Constant patch** — if every `boundaries[k]` collapses to the single point P (all four control points = P), the patch evaluates to P everywhere.

## Acceptance

- All new properties pass 100 cases.
- `make -C tests test` total count grows by the number of new properties.
- README property count refreshed (todo #4).
