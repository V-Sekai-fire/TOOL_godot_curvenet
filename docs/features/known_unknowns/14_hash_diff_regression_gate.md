# 14 — Hash-diff regression gate (ufbx-style)

## Why (current pain)

The trust-topology audit ([AUDIT_TRUST_TOPOLOGY.md](../../AUDIT_TRUST_TOPOLOGY.md))
identified the largest pipeline gap as the absence of an automated
gate between G4 (`scons -j8` + tests pass) and G5 (manual editor
verification on Mire). Today nothing catches "deformer compiles fine,
matches its Lean spec on small fixtures, but produces silently
different output on the Mire rig" — a refactor or compiler change
can drift the deformed geometry without tripping any gate. The
[ufbx](https://github.com/ufbx/ufbx) project has a battle-tested
solution worth adopting wholesale.

## Gall-minimum slice

- **In scope**: a `hash_curvenet` binary that loads a Mire fixture,
  runs `apply_deformation`, FNV-1a-hashes the deformed
  `PackedVector3Array` + `ddm_influences` to a single 64-bit hash.
  Two modes, mirroring `ufbx/test/hash_scene.c`:
  - `hash_curvenet <scene>` → prints the hash.
  - `hash_curvenet --check baseline.hashes.txt` → re-hashes every
    fixture in the baseline and reports mismatches.
- On mismatch, `--dump structured.txt` writes a tagged-tree dump
  (`deformer/positions[v]/x: <hex>`); a `hash_diff.py` clone
  compares two dumps with parent-stack breadcrumbs so the failure
  message points at vertex 1843's `.y`, not "line 5530".
- **Deferred**: image-diff regression (different gate, harder
  setup, higher flake rate); CI / GitHub Actions runner
  integration (separate todo); Hausdorff-distance fallback for
  "geometric equivalence under small drift".
- **Why this slice**: ufbx already proved the architecture works on
  hundreds of FBX fixtures across thousands of CI runs. We just
  port it.

## Files to touch

- `tests/hash_curvenet.c` (new) — main binary, mirrors
  `ufbx/test/hash_scene.c` (~280 LOC adapted).
- `tests/hash_curvenet.h` (new) — FNV state, `ufbxt_hash_pod_imp` /
  `ufbxt_push_tag` / `ufbxt_pop_tag` style API. Crib from
  `ufbx/test/hash_scene.h`.
- `misc/hash_diff.py` (new — copy of ufbx's; ~150 LOC).
- `tests/hash_curvenet.baseline.txt` (new) — committed baseline
  hashes for the Mire fixture set.
- `tests/Makefile` — add `hash_curvenet` target.
- `tests/Mire_5k.scn`, `tests/Mire_81k.scn` — committed fixture
  files (or pointers to existing data files).

## Approach

- Port `ufbx`'s FNV-1a state + tag stack verbatim. Their
  endian-normalised byte hashing handles Linux/macOS/Windows
  reproducibility for free.
- For each `Vec3` in the deformed positions: `ufbxt_hash_real(h,
  v.x); ufbxt_hash_real(h, v.y); ufbxt_hash_real(h, v.z);`.
  NaN routes to a sentinel (`UINT64_MAX`) so NaN-emitting paths
  agree.
- Hash the DDM influence rows too — sparse `(handle_idx, weight)`
  pairs per vertex. Catches bind-step drift, not just runtime
  drift.
- Tagged-tree dump on `--dump` writes:
  ```
  curvenet {
    deformed_positions: [
      [0]: { x: 3ff0000000000000, y: 0, z: 0 }
      ...
    ]
    ddm_influences: [
      [0]: { handle: 3, weight: 3fd5555555555555 }
      ...
    ]
  }
  ```
- `hash_diff.py` compares two dumps and prints context-aware
  failures with parent-stack breadcrumbs.
- **Caveat for us**: ufbx is bit-deterministic because parsing is.
  CG-based solves aren't — small ulp drift across compilers is
  normal. Two options:
  - (a) quantize `Vec3` to 1e-6 before hashing (loses bit-exact
    reproducibility for the same compiler, gains compiler-version
    independence).
  - (b) keep bit-exact, pin the toolchain in CI (Docker / specific
    clang version).
  Recommend (b) for the first cut — tighter regression detection,
  one toolchain version is easy to pin. Switch to (a) if pinning
  becomes onerous.

## Verification

- **Lean**: no change. This is a runtime-output regression gate,
  not a spec change.
- **C++**: `hash_curvenet --check tests/hash_curvenet.baseline.txt`
  passes against a baseline generated from a known-good revision.
  Add to `make -C tests test` so it runs in the standard sweep.
- **GDExtension**: `scons -j8` clean (the new binary is in `tests/`
  and doesn't touch the framework).
- **Manual**: deliberately introduce a one-vertex perturbation in a
  test branch; verify `hash_curvenet --check --dump` produces a
  legible failure message naming the offending vertex.

## Blocks / blocked-by

- **Blocks**: nothing strictly. **Enables**: the full CI workflow
  recommended in [`AUDIT_TRUST_TOPOLOGY.md`](../../AUDIT_TRUST_TOPOLOGY.md)
  Recommendation 1 (CI runner for the algorithmic gates).
- **Blocked-by**: nothing.
- **Related**:
  - [04 sparse bind harvest](04_sparse_bind_harvest.md) — once that
    lands, the bind-step output changes; baseline must be
    regenerated.
  - [03 Quest 3 GPU dispatch](03_quest3_gpu_dispatch.md) — GPU vs
    CPU output may differ; baseline likely needs a per-backend
    suffix.

## Estimated cost

- LOC: ~500 across `hash_curvenet.{c,h}` (~280 + ~120 ported from
  ufbx) + `hash_diff.py` (~150 ported) + Makefile (~30) + baseline
  text + fixture wiring (~50).
- Effort: medium. The architecture is ufbx's, the integration with
  curvenet's deformer is the new bit.
- Risk: low. Reference impl is mature, the surface we're hashing
  is small (positions + influences), and bit-exact pinning to one
  compiler is operationally simple.
