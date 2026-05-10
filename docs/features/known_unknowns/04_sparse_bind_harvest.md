# 04 — Sparse bind harvest

## Why (current pain)

DDM bind harvest at `src/curvenet_deformer_3d.cpp` calls the **dense**
`harmonic_solve::solve_multi`, which builds the full nv×nv `lhs` and
runs `dense::solve_multi`. On the 81 k Mire body that's ~30 s — too
slow for interactive iteration when the artist re-binds (every
topology change, every DDM toggle flip).

## Gall-minimum slice

- **In scope**: replace the dense path in the DDM bind block with
  sparse multi-RHS PCG, using the already-cached `Lh_csr` and
  `LhsM_csr`. One `cg_with_guess` per handle column, warm-started
  against the previous handle's result.
- **Deferred**: parallel handle solves (each handle is independent —
  embarrassingly parallel — but the bind step is amortized so single-
  threaded works for now); GPU-side bind harvest (todo 03 family).
- **Why this slice**: 81 k bind drops from ~30 s to ~1-2 s without
  changing anything algorithmically — same harmonic harvest, same
  output W matrix. Pure perf, no risk.

## Files to touch

- `src/curvenet_deformer_3d.cpp` — replace the
  `harmonic_solve::solve_multi(...)` call inside the DDM bind block
  (`if (use_direct_delta_mush && ...)`) with a per-column loop using
  `sparse::cg_with_guess` (or `incomplete_cholesky::cg_icc_with_guess`
  when the ICC factor is built).
- `tests/bench_ddm_bind_81k.cpp` (new) — benchmark before/after
  bind cost on 81 k Mire body.

## Approach

- For each handle column h ∈ [0, nc):
  - Build per-column RHS: `f_c = e_h` (one-hot at h). Map into
    halfedge-space C·f_c (the existing
    `harmonic_solve::compute_c_fc_matrix` wrapped at k=1 works).
  - Build LHS RHS: `b = -V^T · L_h · C · f_c` (single column, sparse
    SpMV against `Lh_csr` then scatter-V).
  - Solve: `x_v_h = cg_with_guess(LhsM_csr, b, prev_x, max_iter, tol)`.
    Warm-starting from the previous column's solution converges
    fastest when adjacent handles have similar harmonic responses
    (true for nearby curve knots).
  - Stack `x_v_h` as column h of W.
- After all columns: smooth via existing
  `direct_delta_mush::smooth_weights`, sparsify via `sparsify_top_k`,
  store in `rest_cache.ddm_influences`. No change to that downstream
  path.
- Use `LhsM_csr` as the system matrix (same as the per-frame §6
  solve uses), not `Lh_csr` directly — that way the solve is the same
  nv×nv SPD problem the existing CG path is tuned for.
- Tolerance: 1e-7 is plenty for DDM weights (the smoothing step
  diffuses sharp errors anyway).

## Verification

- **Lean**: no change. The bind step is C++ orchestration of already-
  specced primitives.
- **C++**: `tests/bench_ddm_bind_81k.cpp` measures bind time before
  and after. Acceptance: < 3 s on 81 k Mire body, M2 Pro.
- **GDExtension**: `scons -j8` clean.
- **Manual**: in editor, toggle `use_direct_delta_mush` on a 81 k
  mesh and verify the apply step finishes in single-digit seconds
  rather than the previous ~30 s.

## Blocks / blocked-by

- **Blocks**: 11 (multi-character batching benefits from each
  per-character bind being cheap).
- **Blocked-by**: none.

## Estimated cost

- LOC: ~150 across deformer (~120) + bench (~30).
- Effort: small.
- Risk: low. CG infrastructure is mature, ICC opt-in already works,
  warm-start is already wired for the per-frame solve. The only
  novelty is using it at bind time rather than runtime.
