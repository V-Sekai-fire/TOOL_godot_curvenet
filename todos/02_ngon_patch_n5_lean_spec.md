# Cycle 10 #2 — Lean4 spec for N≥5 patch evaluation

**File:** `lean/Curvenet/NgonPatch.lean:7`

## Current state

```
N>=5 -> reserved for mean-value-coords blending (TODO)
```

`Curvenet/MeanValue.lean` already exists with `Vec2`, `regularNgon`, and `meanValueWeights` (committed in `7d9bb2a`).

## Goal

Define an `ngonEvaluate : List BoundaryCurve → Vec2 → Vec3` (or equivalent) that mirrors the C++ formula chosen in todo #1, with `native_decide` corner-recovery checks on a regular pentagon.

## Suggested checks

- For each k ∈ {0..4}: evaluating the pentagon patch at `regularNgon 5 |>.get! k` returns `boundaries[k].c0` (the shared corner) — bit-exact.
- Side-midpoint check (optional): for linear boundary curves, evaluating at the midpoint of side k returns the midpoint of `V_k` and `V_{k+1}` lifted to 3D.

## Dependencies

- Pick the C++ formula first (or co-design with todo #1) so Lean spec and C++ implementation stay in lockstep.
