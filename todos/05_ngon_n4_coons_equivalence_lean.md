# Deferred — N=4 NgonPatch ↔ CoonsPatch equivalence (Lean4)

**File:** `lean/Curvenet.lean:14`

## Current state

```
- N=4 NgonPatch ↔ CoonsPatch equivalence ........... TODO (definitional once NgonPatch is encoded)
```

NgonPatch.lean already encodes the N=4 case via `triPatch`-style canonical reorientation. The Coons machinery is in `Curvenet/CoonsPatch.lean`.

## Goal

State and prove that for any 4 boundary curves with shared corners, the N=4 branch of `NgonPatch` evaluates to the same `Vec3` as the corresponding `CoonsPatch.evaluate` for every `(s, t)`.

Likely shape:

```lean
theorem ngonPatch_n4_eq_coons (b0 b1 b2 b3 : BoundaryCurve) (s t : Float) :
    /- N=4 NgonPatch eval -/ = /- equivalent CoonsPatch eval -/ := by
  rfl  -- or `unfold ... ; rfl` — definitional once unfolded
```

May require `rfl` to close, or instance-level `native_decide` checks if Float `rfl` is too brittle (likely it is — Float arithmetic isn't reducible at `rfl`).

## Priority

**Deferred** — independent of Cycle 10. Pick up after the N≥5 work lands.
