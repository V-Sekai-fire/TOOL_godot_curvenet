# 12 — Test + spec infrastructure

## What it is

Two layers of automated correctness checks, both green, both
required-clean before any commit:

1. **Lean 4 spec layer** — 23 modules under `lean/Curvenet/` with
   258 `native_decide` / `decide` proofs. Each module mirrors a C++
   header in `src/curvenet/`. Each proof exercises the algorithm on
   a concrete small instance — the pattern is "spec + run + check
   answer matches" rather than fully-quantified theorems.
2. **RapidCheck property tests** — 13 test binaries under `tests/`
   with 92 RC properties. Each mirrors a Lean module's invariants and
   adds randomized property tests on the C++ implementation.

Zero `sorry` / `admit` in any Lean file. Zero failing properties.

## Status

internal — used by every contributor pre-commit; not directly
artist-facing. Required-green status enforced by manual verification
(no CI yet).

## Files

### Lean
- `lean/Curvenet/*.lean` — 23 modules
- `lean/Curvenet.lean` — umbrella import
- `lean/lakefile.lean` — build target

### C++ tests
- `tests/test_*.cpp` — 13 RC binaries
- `tests/Makefile` — build + run
- `tests/bench_*.cpp` — benchmarks (run individually for per-mesh
  measurements; the historical `PERF_BASELINE.md` aggregator was
  retired)
- `tests/diag_*.cpp` — diagnostic / one-shot runs

## API surface

Run from a clean tree:

```sh
# Lean spec layer:
cd lean && lake build                # → 26 jobs, zero errors

# C++ unit / property tests:
make -C tests test                   # → 92 RC properties green

# GDExtension build:
scons -j8                            # → framework built, no warnings

# Proof tally:
cd lean/Curvenet && total=0 && \
  for f in *.lean; do \
    total=$((total + $(grep -cE 'native_decide|by decide' "$f"))); \
  done && echo "Total proofs: $total"  # → 258
```

## How it works

- **Lean**: each module file ends with a `namespace XExamples` block
  containing `example : ... := by native_decide` blocks. The Lean
  kernel evaluates the algorithm on the concrete fixture and checks
  the asserted invariant. `Float` equality works via `vecWithinEps`,
  `mat3WithinEps`, `fclose` patterns (Lean doesn't synthesize
  decidable Float equality, but eps-bounded `<` is decidable).
- **RapidCheck**: each `tests/test_*.cpp` defines property functions
  via `rc::check("description", []{ RC_ASSERT(...) })`. Properties
  are randomized over input domains; default 100 cases per property.
  Failures shrink to a minimal counter-example.
- **Test discipline**: a Lean module without proofs is allowed; an
  RC test without properties is not. Algorithms with no invariants
  worth checking (e.g., `Vec3` data struct) get a single trivial
  proof to keep the pattern uniform.

## Cross-references

- Pre-commit verification commands in every feature's "API surface"
  section
- Trajectory of retired families with passing tests-but-failing-
  measurements documented in the
  [archive repo](https://github.com/V-Sekai-fire/TOOL_godot_curvenet_archive)
- The 5 retired modules (multilevel Schwarz, HEM, kernel projection,
  Chebyshev, two-level Schwarz) had clean tests too — proofs verified
  *correctness*, not *fitness*. Both layers are necessary.
