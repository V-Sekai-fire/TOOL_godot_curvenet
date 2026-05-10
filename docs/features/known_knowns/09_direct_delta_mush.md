# 09 — Direct Delta Mush

## What it is

Le-Lewis 2019 DDM, curvenet-adapted. Bake DeGoes22's harmonic
response per handle once at bind time, smooth + sparsify into a
per-vertex sparse influence list, then runtime is a sparse LBS-style
matvec: `pos[v] = Σᵢ W[v, i] · T_i · rest_pos[v]` with `T_i` the
4×4 affine built from the artist's current per-knot frame.
Translates the §6 cost from per-frame (~3-50 ms) to one-shot bind
(~1 s after sparse harvest lands; ~30 s today on 81 k).

## Status

shipping opt-in. Toggle via `CurveNetDeformer3D::use_direct_delta_mush`.
Runtime kernel is CPU; GPU dispatch is
[known unknowns 03](../known_unknowns/03_quest3_gpu_dispatch.md).

## Files

### Lean spec
- `lean/Curvenet/DirectDeltaMush.lean` — 6 native_decide proofs
  (LBS matvec: identity / translation / weighted blend / empty /
  4-handle uniform)
- `lean/Curvenet/DirectDeltaMushBind.lean` — 12 native_decide proofs
  (smoothWeights preserves uniform-half, sparsifyTopK keeps top-K +
  renormalizes, partition-of-unity composition invariants)

### C++ implementation
- `src/curvenet/direct_delta_mush.h` — `Mat4`, `Influences`,
  `apply_transform_4x4`, `lbs_matvec`, `WeightMatrix`, `Adjacency`,
  `partition_of_unity`, `smooth_weights`, `sparsify_top_k`,
  `sparse_row_sum`

### Runtime integration
- `src/curvenet_deformer_3d.cpp` — bind block harvests scalar
  harmonic response per handle, smooths + sparsifies into
  `rest_cache.ddm_influences`. Runtime branch builds per-handle
  `Mat4` from DeGoes22 §3 (`F = F_perp_scale · R_tilt ·
  isolated_segment_gradient`) and dispatches `lbs_matvec` per vertex.

### Tests
- `tests/test_direct_delta_mush.cpp` — 14 RC properties (kernel
  correctness + bind invariants + composition)

## API surface

GDScript-facing properties on `CurveNetDeformer3D`:
- `use_direct_delta_mush : bool` — opt in
- `ddm_top_k : int` (default 4) — sparsify retain count per vertex
- `ddm_smooth_iters : int` (default 3) — Laplacian smoothing passes
  before sparsification

## How it works

**Bind step** (runs once per topology change OR DDM toggle flip):

1. Identity `Fc` (one-hot per handle column, k = nc) drives a single
   `harmonic_solve::solve_multi` → per-vertex W matrix shape nv × nc
   where `W[v, h]` = harmonic response of handle h at vertex v.
2. Sample-promoted vertex rows are overridden to one-hot (samples
   follow their handle exactly).
3. Build vertex-vertex adjacency from `tri_indices`.
4. `smooth_weights(W, adj, ddm_smooth_iters, omega=0.5)` — damped
   Jacobi diffusion.
5. `sparsify_top_k(W_smoothed, ddm_top_k)` → per-vertex sparse
   influence list, renormalized to partition-of-unity.

**Runtime step** (per drag):

1. Build per-handle `Mat4` from current Curve3D state:
   - `F_segment = isolated_segment_gradient(rest_p, rest_q,
     posed_p, posed_q)` — tangent rotation + length scale
   - `R_tilt = Rodrigues(posed_tangent, Δtilt)` — twist around
     tangent
   - `F_perp = s_w · I + (1 − s_w) · t⊗tᵀ` — perpendicular-to-
     tangent scale by `w_posed/w_rest`
   - `F = F_perp · R_tilt · F_segment`
   - `T = [F | posed_p − F·rest_p ; 0 0 0 1]`
2. For each vertex, `lbs_matvec(transforms, infl[v], rest_pos[v])`.
3. Emit deformed mesh.

## Cross-references

- Algebra layer: [06 scaled frames](06_scaled_frames.md)
- Bind solver: [07 solver kernels](07_solver_kernels.md) +
  [08 §6 solve](08_degoes22_solve.md) (the harvest is `solve_multi`
  with identity Fc)
- Authoring UX: [10](10_authoring_ux.md) — all four knot knobs feed
  into the per-handle Mat4
- Quest 3 GPU port:
  [known unknowns 03](../known_unknowns/03_quest3_gpu_dispatch.md)
- Bind perf:
  [known unknowns 04](../known_unknowns/04_sparse_bind_harvest.md)
