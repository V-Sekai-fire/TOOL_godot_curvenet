# 07 — Solver kernels

## What it is

In-house linear-algebra primitives. Dense LU + transpose for small
problems, sparse CSR + matvec + Jacobi-preconditioned CG for the
nv × nv `V^T L_h V` system, ICC(0) factorisation as an opt-in
preconditioner, and the HSC bind-time variant. Welsh-Powell graph
coloring for parallelizing GS-style smoothers. **Eigen, SuiteSparse,
and Mathlib are banned project-wide** — this module is the entire
linear-algebra surface area.

## Status

shipping default — every §6 solve uses the sparse CG path; the bind
step uses dense for now (DDM weight harvest — see
[known unknowns 04](../known_unknowns/04_sparse_bind_harvest.md)).

## Files

### Lean spec
- `lean/Curvenet/DenseLinAlg.lean` — 18 native_decide proofs
- `lean/Curvenet/SparseLinAlg.lean` — 9 native_decide proofs
- `lean/Curvenet/IncompleteCholesky.lean` — 5 native_decide proofs
- `lean/Curvenet/HierarchicalSparsifyCompensate.lean` — 12 proofs
- `lean/Curvenet/GraphColoring.lean` — 8 native_decide proofs

### C++ implementation
- `src/curvenet/dense_linalg.h` — Gaussian elimination with pivoting,
  multi-RHS solve
- `src/curvenet/sparse_linalg.h` — `SparseMatrixCSR`, `spmv`,
  `spmv_multi`, `cg`, `cg_with_guess` (Jacobi-preconditioned)
- `src/curvenet/incomplete_cholesky.h` — `factor`,
  `factor_with_retry` (with `diag_shift` retry on breakdown),
  `cg_icc`, `cg_icc_with_guess`. Includes the inlined
  `detail::zero_mean_in_place` for null-space-aware solves.
- `src/curvenet/hierarchical_sparsify.h` — HSC primitives (greedy
  IS coarsening + Schur compensation)

### Tests
- `tests/test_dense_linalg.cpp`
- `tests/test_incomplete_cholesky.cpp`
- `tests/test_hierarchical_sparsify.cpp`

## API surface

C++-internal. Used by the deformer:
- `sparse::cg_with_guess(A, b, x0, max_iter, tol)` — warm-started CG
- `incomplete_cholesky::cg_icc_with_guess(A, fac, b, x0, max_iter,
  tol, project_kernel)` — PCG with the cached ICC factor

## How it works

- **Dense**: Gaussian elimination with partial pivoting. For systems
  small enough to be dense (e.g., k=9 multi-RHS over a tiny mesh).
- **Sparse CSR + CG**: standard preconditioned CG with Jacobi
  preconditioner (diag(A)⁻¹). Warm-starting is wired through the
  deformer's per-frame solve so adjacent drag frames converge in 1-2
  iters.
- **ICC(0)**: no-fill incomplete Cholesky. ~3× cost per CG iter vs
  Jacobi, but cuts iter count from `~sqrt(κ)` to `~κ^{1/4}` on the
  cot-Laplacian. Lazy-built first time `use_incomplete_cholesky` is
  observed true. `factor_with_retry` does Manteuffel 1980 shifted
  ICC if the no-fill variant breaks down.
- **HSC**: greedy independent-set coarsening + Schur compensation.
  Used at bind for hierarchical sparsification (currently opt-in;
  ICC outperforms HSC in production tests at 81 k).
- **Graph coloring**: greedy Welsh-Powell. Enables shared-nothing
  parallel SGS smoothing and feeds the GPU compute path.

## Cross-references

- Used by every solve in [05 cot-Laplacian](05_cot_laplacian.md)
  and [08 §6 solve](08_degoes22_solve.md)
- ICC opt-in toggled via `CurveNetDeformer3D::use_incomplete_cholesky`
- HSC variants and Schwarz-family preconditioners ruled out across
  ~100 loops — see [`IMPOSSIBILITY.md`](../../IMPOSSIBILITY.md) and
  the
  [archive repo](https://github.com/V-Sekai-fire/TOOL_godot_curvenet_archive)
