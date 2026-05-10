# 08 — DeGoes22 §6 two-stage solve

## What it is

The default per-frame deformation solver from DeGoes22 §4.3.
Stage 1 (Eq. 6a): solve `(V^T L_h V) F_v = -V^T L_h C f_c` for
per-vertex deformation gradients (9-column).
Bridge (Eq. 3): assemble `f_h = V f_v + C f_c`, average over each
cut-face's halfedges to get per-face `F_f`, then build
`y_h[h] = X̆_f F_f^T` for each interior halfedge.
Stage 2 (Eq. 6b): solve `(V^T L_h V) X_v = -V^T L_h (C X_c - y_h)`
for the deformed vertex positions (3-column).

## Status

shipping default — runs on every `apply_deformation` call when
`use_direct_delta_mush` is false. With warm-started PCG and
optional ICC, character-mesh-sized solves take 3-7 ms / 12 RHS on
M2 Pro at 5 k vertices.

## Files

### Lean spec
- `lean/Curvenet/HarmonicSolve.lean` — 5 native_decide proofs
  (Eq. 6a on triangleWithSample with k=1, k=3, k=9)
- `lean/Curvenet/DeformSolve.lean` — 7 native_decide proofs
  (Eq. 6a + 6b end-to-end: translation, rotation, uniform scale,
  composition)

### C++ implementation
- `src/curvenet/harmonic_solve.h` — `compute_cfc`,
  `compute_c_fc_matrix`, `solve_scalar`, `solve_multi`
- `src/curvenet/deform_solve.h` — `compute_fh`,
  `average_over_faces`, `compute_yh`, `solve_deformation`

### Runtime integration
- `src/curvenet_deformer_3d.cpp::apply_deformation` runs the
  multi-RHS sparse PCG path with warm start across frames

### Tests
- `tests/test_deform_solve.cpp`
- `tests/test_deformer_pipeline_identity.cpp` — end-to-end identity
  pipeline: identity Fc + rest-pose Xc → output equals input

## API surface

C++-internal end-to-end pipeline. The artist surface is the
deformer's `apply_deformation` GDScript method, which runs this path
unless DDM is opted in.

## How it works

- Stage 1 builds `Fc = identity per sample` (9-col) and the
  per-halfedge `C·Fc` matrix; SpMV-multi against `Lh_csr`; scatter
  via `apply_vt`; sparse multi-RHS PCG against `LhsM_csr` →
  per-vertex deformation gradients `Fv`.
- Bridge step: `Fh = V·Fv + C·Fc` per halfedge. Average over each
  cut-face's halfedges to get per-face `Ff`. Build
  `yh[h] = X̆_f · Ff^T` where X̆_f is the rest face polygon.
- Stage 2: build `(C·Xc - yh)`; SpMV-multi; scatter; sparse multi-RHS
  PCG → deformed positions `Xv`. Sample-promoted vertex slots return
  0 (degenerate row); the caller overlays `Xc` for those vertices in
  the emit step.
- Warm-start: `prev_Fv`, `prev_Xv`, `prev_solve_valid` carried in the
  rest cache. On any continuous drag, CG converges in 1-2 iters.

## Cross-references

- Drives the runtime path when DDM is off
- The DDM bind harvest (in [09](09_direct_delta_mush.md)) reuses
  `harmonic_solve::solve_multi` with an identity Fc to capture
  per-handle harmonic responses
- Pre-100-loops trajectory of preconditioner choices archived in the
  [archive repo](https://github.com/V-Sekai-fire/TOOL_godot_curvenet_archive)
