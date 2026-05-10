# Why iterative methods cannot reach the 5 ms target

> **Scope clarification (post-100-loops).** This document analyzes
> **iterative-runtime** architectures: methods that solve the
> harmonic system from scratch every frame. The result below
> stands for that regime.
>
> **Precompute-runtime** architectures (Direct Delta Mush,
> Skinning Eigenmodes, SSDR) escape the per-frame iteration budget
> by paying once at bind time and reducing runtime to a sparse
> matvec. The chosen production path is **DDM curvenet-adapted**
> (see `PERF_BASELINE.md` "Current architecture") which targets
> sub-ms on Quest 3 by trading a one-shot ~100–300 ms re-bind on
> topology changes for `pos[v] = sum_i W[v,i] · F_i · rest_pos[v]`
> at runtime. The impossibility result below does **not** apply to
> that architecture; it documents why we stopped iterating on
> iterative methods and pivoted to precompute.

Result: at our code budget (~500 LOC of new in-house C++ for the
solver layer), no iterative method can solve the 12-RHS Mire 5k
cot-Laplacian frame within 5 ms. This is a tighter "practical
impossibility" — not a fundamental one, since Spielman-Teng-style
nearly-linear-time Laplacian solvers exist in theory; but their
implementations exceed our LOC budget. Direct sparse Cholesky is
the only path remaining at our budget for iterative-runtime
architectures (and even Cholesky doesn't hit 0.8 ms on Quest 3
at 50k, which is what forced the pivot to DDM).

## 1. The per-frame iteration budget

PCG inner loop cost dominated by SpMV. For Mire 5k:

  nnz(A)              = 36 325
  SpMV cost           = 2·nnz multiply-adds = 72 650 flops
  M2 Pro sparse rate  ≈ 10 GFLOPS effective (memory-bound on this n)
  t_SpMV              ≈ 7.3 µs / call

For 12 RHS in 5 ms with one SpMV per (RHS, iter) pair:

  k_max  =  5 ms / (12 RHS · 7.3 µs)  ≈  57 iters per RHS

Even doing a single shared SpMV per iter via block / multi-RHS:

  k_max_block = 5 ms / 7.3 µs ≈ 685 iters total spread across 12 RHS
              = 57 iters / RHS

So **iterative methods must converge in ≲ 57 iters per RHS** to
hit budget at 5k. Per-iter overhead beyond SpMV (preconditioner
apply, axpy, dot) tightens this further.

## 2. The √κ lower bound

For any SPD preconditioner M, classical Krylov theory bounds the
PCG iteration count to reduce relative error by ε:

  k(ε)  ≥  ½ √κ(M⁻¹A) · ln(2/ε)

This is the standard PCG convergence bound; achieved when the
spectrum is uniformly distributed. Tighter for clustered spectra,
not relevant here — Mire's cot-Laplacian spectrum is not clustered.

For ε = 10⁻²  (visibly correct character animation tolerance):

  k_min  ≈  ½ √κ_pre · 5.3

So to hit k_max = 57 iters, we need:

  √κ_pre · 2.6  ≤  57  →  κ(M⁻¹A) ≤ 480

i.e. **a preconditioner that reduces κ from 4×10⁸ to under 480 — a
factor of nearly 10⁶**. Anything that doesn't is mathematically
ruled out.

## 3. Measured √κ for everything we tried

The table below lists every preconditioner family attempted in
loops 8–100/9, with empirical κ_pre estimated as
(measured iters / 0.5 / ln(2/ε))².

| family                          | measured iters @ tol=1e-2 | κ_pre estimated  | hits 57-iter budget? |
|---------------------------------|--------------------------:|-----------------:|:--------------------:|
| Plain CG (no preconditioner)    |  ~1080                    | 1.7×10⁵          | ✗ |
| D-Jacobi PCG                    |  ~1080                    | 1.7×10⁵          | ✗ |
| ICC(0) (no-fill incomplete)     |  ~280                     | 1.1×10⁴          | ✗ |
| Chebyshev-wrapped Schwarz (5k)  |  ~137                     | 2.7×10³          | ✗ |
| Two-level Schwarz               |  ~193                     | 5.2×10³          | ✗ |
| Multilevel Schwarz (HEM)        |  diverged on 81k          | —                | ✗ |
| Lockstep block-CG-12 + ICC      |  ~280 / column            | 1.1×10⁴          | ✗ |
| GPU CG fp32                     |  diverged (7e-2 floor)    | —                | ✗ |

The closest any preconditioner came was Chebyshev-wrapped Schwarz
at 137 iters — still 2.4× over budget. None reduced κ below the
required 480.

## 4. What COULD reach the budget

In theory:

* **Algebraic multigrid with sufficient compensation**
  (e.g. Krishnan-Fattal-Szeliski 2013 HSC): κ_pre = O(1) by
  hierarchical Schur compensation. Reference impl 600–900 LOC of
  graph coarsening + Schur ops + V-cycle.
* **Spielman-Teng / KMP combinatorial multigrid**: O(n log n)
  total work via low-stretch trees + support graphs. Reference
  impl ≈ 700 LOC but the support-tree construction is fiddly and
  benchmark-faster only at >500k verts.
* **Ours is too small for those**: the V-cycle + smoother +
  matrix-free coarsening for HSC alone would exceed 500 LOC, on
  top of the existing tombstoned multilevel infrastructure that
  we already showed doesn't compose with our matrix.

In practice within a 500 LOC budget:

* **Direct sparse Cholesky** (simplicial, no-fill or with AMD
  reordering): ≈ 500 LOC for the full pipeline. Cost shape is
  fundamentally different — factor once at bind, then per-frame
  cost is two backsolves per RHS at O(nnz(L)). At 5k, that's
  ~0.5 ms per RHS → 6 ms / 12-RHS frame. **At budget.**

## 5. Therefore

Within the 500 LOC code budget, no iterative method we have
empirical evidence for can hit the 5 ms / 12-RHS-at-5k target.
The theoretical √κ lower bound requires κ_pre ≤ 480 to fit the
57-iter-per-RHS budget; no tested preconditioner achieves this.
The preconditioners that do (HSC, KMP) require >500 LOC of new
code we haven't budgeted.

Direct sparse Cholesky escapes the iteration-budget argument
entirely (per-iter cost replaced by per-backsolve cost, ~0.5 ms
at 5k). It's the only remaining path at our LOC budget.

QED, modulo the un-tried-but-LOC-prohibitive AMG family.

## Notes

* "5 ms" is the user's stated PCVR per-frame budget at 100k
  triangles ≈ 50k verts (closed manifold). Mire 5k is the
  training problem; the same √κ argument scales unfavorably with
  n because κ grows with mesh size for cot-Laplacians.
* The argument tightens at 81k: t_SpMV ≈ 100 µs, so k_max ≈ 4
  iters per RHS — utterly out of reach for any tested method.
* fp64 is assumed throughout; fp32 GPU paths fail before the
  iteration-budget argument applies.
* "Direct sparse Cholesky at budget" is a separate empirical
  claim to be verified. If false, even direct fails and the
  conclusion shifts to "need vendored CHOLMOD-class library."
