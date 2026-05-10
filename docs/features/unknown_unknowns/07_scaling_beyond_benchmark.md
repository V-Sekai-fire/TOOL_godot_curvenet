# 07 — Scaling beyond benchmarked size

## What we don't know

Largest tested mesh is **Mire body at 81 k vertices**. Behavior at
200 k / 500 k / 1 M is unknown. Pre-100-loops trajectory ruled out
several iterative preconditioners *because* they failed at 81 k —
that retired-trajectory record now lives in the
[archive repo](https://github.com/V-Sekai-fire/TOOL_godot_curvenet_archive).
We don't know how the *current production path* (sparse PCG + ICC
opt-in + DDM) holds up beyond 81 k.

## What might break

- ICC factorisation: ~28 s on 81 k, untested on 200 k. Empirically
  ICC nnz scales sub-quadratically with nv but the constant is
  hardware-dependent.
- DDM bind: dense `solve_multi` is O(nv³) — already 30 s on 81 k.
  At 200 k that's ~470 s ≈ 8 minutes. Unusable until
  [known unknowns 04 (sparse bind harvest)](../known_unknowns/04_sparse_bind_harvest.md)
  lands.
- Memory: `LhsM_csr` at 81 k is ~50 MB. At 1 M it's ~600 MB. ICC
  factor adds another ~2× nnz. Per-deformer memory scales linearly
  with nv; multi-deformer scenes hit aggregate ceilings fast.
- Numerical: cot-Laplacian condition number scales with `(h_max /
  h_min)²` — fine on uniform meshes, blows up on irregular
  remeshings (which character meshes often have). Beyond 81 k there
  may not be a working preconditioner family — same impossibility
  result that ruled out Schwarz at 81 k.
- Bind-step latency past 1 minute is artist-unfriendly even for
  one-shot. Scenes with 5-character casts at 200 k each become
  unworkable.

## How we'd find out

- Subdivide the Mire body to 200k / 500k via meshoptimizer. Run the
  full pipeline. Measure bind, runtime, accuracy (compared to the
  81 k baseline through linear interpolation if possible).
- Stress test on a real scanned mesh (lots of artifacts at high
  vertex count).
- Profile memory over the bind step on a 1 M synthetic mesh; identify
  the dominant data structure and its scaling factor.

## Mitigation if it breaks

- Hard cap `vertex_count` at the bind step with a clear error.
  Better than silent failure.
- Spatial decomposition: split the mesh into meshlets (~10 k each),
  bind each independently, blend at boundaries. The
  [archive repo](https://github.com/V-Sekai-fire/TOOL_godot_curvenet_archive)
  has trajectory data on meshlet approaches that didn't work at 81 k
  *for iterative preconditioners*; meshlet-DDM is a different problem
  and may be tractable.
- A "scale tier" property on the deformer: small / medium / large
  switches between dense / sparse / GPU paths.
- Measurements logged into the project's perf record (PERF_BASELINE
  was retired alongside the trajectory docs; new perf measurements
  live in commit messages and benchmark output).
