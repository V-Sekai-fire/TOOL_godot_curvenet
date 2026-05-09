# Cycle 10 #1 — C++ N≥5 patch evaluator

**File:** `src/curvenet/ngon_patch.h:58`

## Current state

The `NgonPatch::evaluate` dispatcher returns `NaN` for `N >= 5`:

```cpp
// TODO(N>=5): Hormann-Floater mean-value coordinates over a regular N-gon domain.
const double nan_val = std::nan("");
return Vec3{ nan_val, nan_val, nan_val };
```

The supporting math is already in place:
- `src/curvenet/mean_value.h` — `Vec2`, `regular_ngon`, `mean_value_weights` (Floater 2003 tangent-half-angle form, vertex special case).
- `tests/test_mean_value.cpp` — 4 RapidCheck properties green.

## Goal

Replace the NaN branch with a transfinite N-sided patch evaluator that uses `mean_value_weights` against `regular_ngon(n)` to blend the N boundary curves.

## Suggested formula

For each side `k` (between V_k and V_{k+1}):
- Side blending function `Φ_k(p) = λ_k(p) + λ_{k+1}(p)`, where `λ` are mean-value weights at the regular-N-gon vertices.
- Side parameter `s_k(p)` — projection of p along side k of the regular N-gon, clamped to `[0, 1]`.
- Ribbon contribution `R_k(p) = boundaries[k].evaluate(s_k(p))`.

Then `S(p) = Σ_k Φ_k(p) · R_k(p)`. (Σ_k Φ_k = 2 · Σ_k λ_k = 2, so divide by 2 — or normalize.)

At a vertex V_m: λ_m = 1, others = 0, so `Φ_{m-1}(V_m) + Φ_m(V_m) = 2`, and `R_{m-1}(V_m) = c3_{m-1} = c0_m = R_m(V_m)`. Recovery is clean.

## Acceptance

- The `NgonPatch::evaluate(s, t)` API is unchanged. Decide what `(s, t) ∈ [0, 1]²` maps to in the regular-N-gon domain (one option: bilinear map of `[0, 1]²` onto a 2-vertex chord plus interior point — or just take `(s, t)` as the 2D query directly when `N ≥ 5`).
- All existing tests still pass.
- New property tests (see todo #3) green.
