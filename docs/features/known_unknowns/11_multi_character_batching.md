# 11 — Multi-character batching

## Why (current pain)

Each `CurveNetDeformer3D` node carries its own `RestCache` with full
sparse matrices, ICC factors, DDM weight rows. A scene with 50
characters duplicates these structures 50 times in CPU and GPU
memory. Pipelines targeting crowd shots or many-NPC games would hit
memory and bandwidth ceilings even though most characters share rest
geometry. Low/research priority gap.

## Gall-minimum slice

- **In scope**: project-level `CurveNetPool` singleton that owns
  shared `RestCache` instances keyed by mesh content hash. Per-deformer
  state shrinks to a pointer into the pool plus per-instance handle
  positions / DDM weight rows.
- **Deferred**: GPU-side instanced dispatch (depends on todo 03
  landing first); cross-frame deduplication (e.g. all NPCs in one
  city level → one dispatch).
- **Why this slice**: solves the memory problem without rewriting
  the runtime path. Each deformer still calls `apply_deformation`,
  but the heavy state is shared.

## Files to touch

- `src/curvenet_pool.h` (new) — singleton holding `unordered_map<hash,
  shared_ptr<RestCache>>`.
- `src/curvenet_pool.cpp` (new) — implementation; lifecycle hooks tied
  to `register_types` for the GDExtension's init/teardown.
- `src/curvenet_deformer_3d.{h,cpp}` — `RestCache` becomes a shared
  pointer fetched from the pool. Per-instance fields move to a smaller
  `InstanceState` struct (handle positions, prev solver iterates,
  DDM weight rows specific to this character's curves).
- `tests/test_curvenet_pool.cpp` (new) — RC properties: same hash
  shares state; different hashes don't; teardown correctly reference-
  counts.

## Approach

- Hash the source mesh content (positions + tri_indices) at bind
  time. If pool already has a `RestCache` for that hash, share it.
- The RestCache split:
  - **Shared**: `positions`, `tri_indices`, `cut_mesh`, `Lh_csr`,
    `LhsM_csr`, ICC factor, mesh-wide DDM weight matrix template.
  - **Per-instance**: handle positions per curve, `col_rest_pos`,
    `col_input_handle`, `prev_Fv`/`prev_Xv`, DDM weight rows
    materialized for *this* character's curve set.
- Per-character DDM weight rows: even with the shared mesh, different
  characters have different curve placements → different sample
  promotions → different W. The shared part is the cot-Laplacian +
  mass; the per-instance part is the harmonic harvest output.
- Pool entries are reference-counted via `shared_ptr`; the last
  deformer to release a hash drops the entry.

## Verification

- **Lean**: no change. The pool is C++ orchestration.
- **C++**: ~5 RC properties on the pool's reference-counting +
  hash-based dedup.
- **GDExtension**: `scons -j8` clean. Add a smoke test scene with
  many character instances.
- **Manual**: instantiate 50 deformers in a scene, verify only one
  set of matrices in memory (use Godot's monitor or a custom
  diagnostic).

## Blocks / blocked-by

- **Blocks**: nothing.
- **Blocked-by**: nothing strictly. **Benefits from** todo 04
  (sparse bind harvest) — the per-character bind step still has to
  run for new character hashes, so faster bind = faster scene load.

## Estimated cost

- LOC: ~700 across pool (~250) + deformer refactor (~300) + tests
  (~150).
- Effort: large.
- Risk: medium. The deformer refactor is the largest single change;
  needs careful split between shared and per-instance state without
  breaking the existing single-character path.
