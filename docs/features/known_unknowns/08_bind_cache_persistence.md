# 08 — Bind cache persistence

## Why (current pain)

`RestCache` is rebuilt on every Godot session's first
`apply_deformation`. On 81 k Mire body that's ~30 s of latency the
artist eats every time they reopen the scene — even when nothing has
changed since save. Medium-priority gap in
[`docs/FEATURES.md`](../docs/FEATURES.md#medium-priority).

## Gall-minimum slice

- **In scope**: serialize `RestCache` to a sidecar
  `<scene>.curvenet_cache.bin` file. Load it on first
  `apply_deformation` if the source-mesh + curves hash matches.
  Otherwise rebuild and overwrite.
- **Deferred**: project-wide cache pool (todo 11); incremental cache
  updates (any topology change → full rebuild stays).
- **Why this slice**: orthogonal to algorithm. Pure I/O. Easiest
  ~30 s → ~1 s win for the desktop authoring workflow.

## Files to touch

- `src/curvenet/cache_serialize.h` (new) — pure binary blob layout
  for `RestCache` fields. Stable enough that older caches can be
  rejected via a magic+version header rather than crash.
- `src/curvenet_deformer_3d.cpp` — bind step: try `load_cache()`
  before rebuilding. After a successful build, call `save_cache()`.
  Cache path = scene path with `.curvenet_cache.bin` suffix.
- `tests/test_cache_serialize.cpp` (new) — RC properties: round-trip
  small `RestCache` instances; rejection of mismatched hash; rejection
  of bad magic / version header.

## Approach

- Define a versioned binary format:
  ```
  magic[8]   = "CRVNT001"
  source_hash[8]
  nv[4], nh[4], nc[4]
  positions[nv * 3 * 8]
  tri_indices[len * 4]
  cut_mesh.vertex_kind[nv * sizeof(CutVertexKind)]
  cut_mesh.segment_of_halfedge[nh * 4]
  Lh_csr.row_ptr / col_idx / values
  LhsM_csr same
  ddm_built[1], ddm_top_k[4], ddm_smooth_iters[4]
  ddm_influences (CSR-flat)
  rest_curve_knots / tilts / widths
  ```
- Write little-endian; check on load.
- `source_hash` is the same `std::uint64_t` already computed in the
  rebuild block. Mismatch → reject and rebuild.
- ICC factor is **not** persisted — it's lazy-built at first opt-in
  and rebuilds in ~1 s on 81 k anyway. Cache only the deterministic
  bind output.
- Use `Godot::FileAccess` for the I/O (handles sandboxing,
  platform paths) but keep the format pure C++ structs so the
  serializer is testable without godot-cpp.

## Verification

- **Lean**: no change.
- **C++**: ~5 RC properties on round-trip + rejection cases.
- **GDExtension**: `scons -j8` clean.
- **Manual**: open a 81 k scene, save, close, reopen. First apply
  should be < 1 s instead of ~30 s. Modify the curve, save, reopen —
  cache invalidates and rebuild fires.

## Blocks / blocked-by

- **Blocks**: nothing.
- **Blocked-by**: nothing.

## Estimated cost

- LOC: ~250 across cache_serialize.h (~150) + deformer (~50) + tests
  (~50).
- Effort: medium.
- Risk: low. Versioned format header lets us iterate without breaking
  saved scenes; if the format changes, old caches are rejected and
  rebuild from scratch — no data corruption surface.
