# DeGoes22 deformer — CPU baseline

Wall-clock numbers for `src/curvenet_deformer_3d.cpp::apply_deformation`
on CPU. Run via:

```sh
make -C tests bench    # builds tests/bench_deform.cpp at -O3 + runs it
```

Tracks three workloads: synthetic plane (smooth, well-conditioned),
synthetic plane via dense LU (regression check for the historical
path), and a real character mesh (Mire Quest body, 5485 verts).

## Real character mesh

`Mire body` is loaded from `tests/mire_body_data.h` (baked from
`demo/character/MireQuest.blend` via Blender MCP). Source:
[V-Sekai-fire/mesh-mille-mire-feuille](https://github.com/V-Sekai-fire/mesh-mille-mire-feuille).
Curvenet is a single closed-loop with N hand-picked sample points
projected to the closest body vertices.

### Current — robust path (Sharp & Crane 2020 mollification)

| label                       | nv    | nh     | nc | bind ms | frame ms  | frames/s |
|-----------------------------|-------|--------|----|---------|-----------|----------|
| Mire body, 4-sample loop    |  5485 |  30856 | 4  |       9 |        91 |     11   |
| Mire body, 8-sample loop    |  5485 |  30856 | 8  |       9 |        90 |     11   |
| Mire body 2× subdivided     | 81613 | 481840 | 4  |     **166** | **~80000** | **0.01** |

The 81k row is the actual PCVR-target measurement (81,613 verts, 159k
tris, baked from `MireQuest.blend` after 2× edge-subdivide). The
working monolithic system is **still 5000× short** of the 11 ms /
90 FPS budget at solve time. The 102-sec bind bottleneck has been
fixed:

* **Bind: 102000 ms → 166 ms (615× faster).** The previous bind
  iterated `face_loop(m, face_id)` once per face, and each call did
  a linear scan over all `nh` halfedges to find the face's start.
  At 81k verts that's 159k × 482k ≈ 76 billion ops just to set up
  the assembly. Replaced with `all_face_loops(m)` — a single O(nh)
  pass that builds the loops for every face up-front. `std::map`
  COO accumulation also replaced with a flat `std::vector<CooEntry>`
  + sort+merge, which is `coo_to_csr`. Both fixes are independent
  of solver choice and benefit every assemble call site.

* **Solve = 163 sec for 12 RHS at 81k (still architectural).**
  Instrumented CG (`sparse::cg_diag`) reports **20,668 iters at
  0.655 ms each per RHS**. Per-iter cost is fine; iter count is the
  problem. κ ≈ 4×10⁸ → √κ ≈ 20k matches the measurement. Even with
  a 10× preconditioner improvement (incomplete Cholesky / multigrid)
  the monolithic CPU path lands at ~17 sec/frame — 1500× over 11 ms.

* **Meshlet CPU path measured at 81k.** `diag_meshlet_pcg_chol_70k`
  on the same mesh: 368 meshlets, bind 766 ms (build + Cholesky
  factor every local), per outer iter 64.7 ms. Convergence rate
  ~0.97/iter once past the early phase, so ~670 outer iters to
  reach 1e-9 → ~43 sec/frame. ~4× better than monolithic but
  still 4000× over budget.

The CPU has been stretched as far as Gall's-law evolution allows.
GPU compute is the only path that fits 11 ms / 90 FPS at 81k. The
existing `tests/gpu_cg_multi_solver.h` is validated on synthetic
SPD systems and ready to be wired against meshlet locals.

The 5k bind drop (421 ms → 9 ms) is a 47× win on a path that
was already fast enough; the win mostly matters at the 70k+ scale
where the quadratic was the difference between "minute-long bind"
and "interactive."

The meshlet decomposition path (per-meshlet ~256×256 matrices solved
independently with cached Cholesky) sidesteps both — per-meshlet bind
is small enough that `std::map` is fine, per-meshlet cold-CG is fast
on a small system. Tracked under `tests/diag_meshlet_*.cpp` and
extrapolated separately.

5485-vert numbers are the deployment-validated CPU ceiling and what
the demo scene currently ships.

### Before — non-robust cot (broken on real meshes)

Until Sharp & Crane 2020 intrinsic mollification landed, the
embedding-based cot in `polygon_laplacian.h` produced `±∞` off-
diagonal entries on this mesh because cross-product magnitudes
underflow on near-collinear edges. CG never converged:

| label                       | nv   | nh    | nc | bind ms | frame ms | frames/s |
|-----------------------------|------|-------|----|---------|----------|----------|
| Mire body, 4-sample loop    | 5485 | 30856 | 4  |  293    | **5779** | **0.2**  |
| Mire body, 8-sample loop    | 5485 | 30856 | 8  |  290    | **5782** | **0.2**  |

The fix (commit landing this section): replace cot-from-embedding
with cot-from-edge-lengths after intrinsic mollification — pad every
edge by ε so the strict triangle inequality holds with margin δ at
every corner, then compute cot via law-of-cosines + Heron. Now finite
and convergent. **63× speedup at the same vertex count.**

## Synthetic plane — sparse + warm-start (the runtime path)

| N×N  | `nv`   | `nh`   | bind (ms) | frame avg (ms) | frames/s |
|------|--------|--------|-----------|----------------|----------|
|  10² |    100 |    522 |       0.3 |           0.07 |  13881   |
|  15² |    225 |   1232 |       0.9 |           0.24 |   4096   |
|  20² |    400 |   2242 |       2.3 |           0.64 |   1568   |
|  25² |    625 |   3552 |       4.9 |           1.43 |    700   |
|  30² |    900 |   5162 |       9.7 |           2.77 |    361   |
|  40² |   1600 |   9282 |      29.0 |           8.09 |    124   |
|  50² |   2500 |  14602 |      68.1 |          18.98 |     53   |
|  70² |   4900 |  28842 |     256.3 |          69.05 |     15   |

These numbers describe a smooth, well-conditioned 2D Laplacian on a
unit-square plane with corner samples. They are **not** representative
of real character meshes — see the row above.

## Synthetic plane — dense LU (kept for regression checking)

| N×N  | `nv`  | `nh`  | bind (ms) | frame avg (ms) | frames/s |
|------|-------|-------|-----------|----------------|----------|
|  10² |   100 |   522 |        41 |           4.35 |    230   |
|  15² |   225 |  1232 |       466 |          25.54 |     39   |
|  20² |   400 |  2242 |      2797 |          87.44 |     11   |
|  25² |   625 |  3552 |     11962 |         215.98 |      5   |
|  30² |   900 |  5162 |     38579 |         458.06 |      2   |

Both columns scale cubically. Dense path stops at 900 verts on
memory; sparse takes over from there.

## Benchmark scope

Included in the per-frame timing:

* assembling `C·F_c` and `C·X_c` per-halfedge constraint matrices
* `Lₕ · (C·F_c)` and the `Vᵀ·…` aggregation (Eq. 6a RHS)
* solve against the cached LHS (LU back-sub for dense, CG for sparse,
  9 columns then 3)
* the bridge: `compute_fh`, `average_over_faces`, `compute_yh`

Excluded from per-frame:

* the SurfaceTool / ArrayMesh emit (Godot-side, hard to bench in
  isolation)
* the curvenet_builder ε-merge and surface_projection (run only at
  bind time in the runtime)

The `bind` column includes everything from `from_triangles` through
the LHS assembly (and LU factorisation for the dense row). Real
`apply_deformation` also runs the curvenet build and sample promotion
at bind, which adds a small constant.

## Chebyshev acceleration on the meshlet Schwarz path

`tests/diag_meshlet_chebyshev_5k.cpp` wraps the per-meshlet
multiplicative-Schwarz outer iteration with Wang 2015's Chebyshev
semi-iterative recurrence (`src/curvenet/chebyshev_accel.h`,
mirroring `lean/Curvenet/ChebyshevAccel.lean`). On the 5k Mire
body with synthetic SPD-projected RHS, tol 1e-9:

| Configuration                   | Outer iters to 1e-9 |
|---------------------------------|---------------------|
| Plain multiplicative Schwarz    | ~2100 (extrap)      |
| Chebyshev ρ = 0.95              |  274                |
| Chebyshev ρ = 0.97              |  197                |
| **Chebyshev ρ = 0.98**          | **137**             |
| Chebyshev ρ = 0.985             |  155                |
| Chebyshev ρ = 0.99              |  189                |
| Chebyshev ρ = 0.995             |  271                |

**~15× outer-iter reduction** at the optimal ρ. Plain Schwarz
contracts at 0.998/iter (measured), so its asymptote is ~2100 iters
to 1e-9; Chebyshev with ρ=0.98 contracts at ~0.85/iter. The
optimum sits slightly *below* the true contraction rate, which
makes Chebyshev robust to spectral-radius mis-estimation.

The 30-line `chebyshev_accel::accel` wrapper is the smallest
evolution from the working Schwarz path that gets a 15× speedup
without changing the underlying meshlet decomposition. This is
the "small surgical fix" Gall's law calls for, applied to the
meshlet-Schwarz architecture.

## How the robust path works

`src/curvenet/robust_laplacian.h` mirrors
`lean/Curvenet/RobustLaplacian.lean` and implements §4 of
Sharp & Crane SGP 2020:

1. **Edge lengths from the embedding.** For each triangle (a, b, c)
   of every cut-face fan-triangulation, compute `eA = |b-c|`,
   `eB = |a-c|`, `eC = |a-b|`.
2. **Per-triangle ε.** For every corner the strict triangle
   inequality reads `a + b > c + δ` where δ is a small safety
   margin. Adding ε to every edge changes the inequality to
   `(a+ε) + (b+ε) > (c+ε) + δ` ⇔ `ε > δ + c - a - b`. Per-triangle
   required ε is the max over the three corners.
3. **Global ε.** Take the max per-triangle ε across the whole
   mesh — using one consistent ε keeps shared-edge contributions
   in agreement.
4. **Cot from mollified lengths.** Law of cosines + Heron's:
   `cot θ = (e_adj1² + e_adj2² - e_opp²) / (4·area)`. With the
   mollified lengths the area is bounded away from zero, so cot
   is finite.

We use δ = 1e-5 × mean edge length of the input mesh. On the Mire
body that's 2.1e-7. The cot Laplacian on the resulting mesh has
finite entries (off-diag dynamic range 10¹¹ vs `±∞` before) and CG
converges.

## Dead ends (loops 8 → 100/1 → 100/2)

Three rounds of preconditioner work all hit the same plateau at
L_inf residual ~3.7 on the 81k Mire cut-mesh Laplacian. Files
remain in tree with TOMBSTONE headers:

| Loop      | Hypothesis                                     | Verdict                                          | Tombstoned files |
|-----------|-------------------------------------------------|---------------------------------------------------|------------------|
| 8         | Recursive multilevel beats 2-level on 81k      | False — 5-level stalls at 3.7                     | `multi_level_schwarz.{h,lean}` |
| 100/1     | HEM connectivity-aware coarsening helps        | False — 7-level HEM stalls at 3.7                 | `heavy_edge_matching.{h,lean}` |
| 100/2     | 1D constant null-space drift is the cause      | False — `max│row_sum│` = 1.16e-10 already         | `kernel_projection.{h,lean}` |

The smoking-gun diagnostic is `tests/diag_70k_cg_baseline.cpp`.
Plain unpreconditioned CG on the 81k matrix converges to ‖r‖² =
3.6e-10 in 200,000 iters (133 s). Every preconditioner above is
**worse than no preconditioner**. The actual cause is in the
matrix itself:

  diag range = [8.2e-2, 1.1e+6]   (7 orders of magnitude)

Galerkin coarsening inherits this range. The Jacobi smoother
`omega · D^{-1} · r` is useless when D spans 7 decades — small-D
rows are over-relaxed, large-D rows are under-relaxed. Plain CG's
implicit per-iter scaling at least homogenizes; discrete Jacobi
sweeps don't.

The `two_level_schwarz` and `chebyshev_accel` modules also carry
TOMBSTONE headers — they pass their unit tests and converged at
5k, but every 81k variant of either stalls at the same plateau.

Don't reuse any of these on a real mesh until the smoother is
swapped for one robust to wide diagonal ranges (sym-GS, or
D^{-1/2}·A·D^{-1/2} symmetric scaling per level), and that
smoother is verified in isolation on the 81k matrix first.

The reproducer for the stall is `tests/diag_multi_level_schwarz_70k`.
The earlier 70k Chebyshev and 2-level Schwarz diags were removed
as redundant.

### Live candidate ✓ (loop 100/3 — ICC(0))

Test gate passed. **Shifted ICC(0)-PCG** (Manteuffel 1980 diagonal
shift retry) clears the 81k stall:

| Mesh   | nv    | nnz    | iters  | wall  | vs prior baseline                          |
|--------|------:|-------:|-------:|------:|--------------------------------------------|
| 5k     | 5485  | 36325  |    351 | 0.05 s | vs Schwarz 168 iters / 2.4 s — **48× wall** |
| 81k    | 81613 | 563389 |  2,478 | 3.65 s | vs plain CG 200k iters / 133 s — **38× wall, 81× iters** |

5k needs a non-zero diagonal shift to factor (no-fill ICC(0) on
the 5k cot-Laplacian breaks down with a non-positive pivot at the
first try; retry loop hits a working shift quickly). 81k factors
clean at shift = 0. Factorisation cost: 28 s once at bind time
on 81k, amortised across all subsequent solves.

This is the first preconditioner that beats plain CG on 81k since
the project began. Six earlier attempts (1-/2-/3-/5-/7-level
Schwarz; HEM; kernel projection; Chebyshev) were all worse than
no preconditioner — see "Dead ends" above and the TOMBSTONE
headers in `src/curvenet/{two_level,multi_level,heavy_edge_matching,
kernel_projection,chebyshev_accel}.h`.

Reproducer: `make -C tests diag_70k_icc_pcg`.

Fallback candidates (still on deck if ICC's per-iter cost trips
us up at higher nv): **Lan 2025 JGS2** (`Lan2025JGS2`,
[PDF](https://wanghmin.github.io/publication/lan-2025-jgs/Lan-2025-JGS.pdf))
— second-order Jacobi/GS targeting the stiff-system regime.
**Lan 2023 stencil descent** (`Lan2023StencilDescent`,
[PDF](https://wanghmin.github.io/publication/lan-2023-sos/Lan-2023-SOS.pdf))
— per-stencil second-order descent.

## Multi-level Schwarz (loop 8 + loop 100)

Generic multilevel V-cycle on top of the meshlet Schwarz smoother.
Hierarchy auto-built until top-level size is below
`TOP_THRESHOLD = 16`. Galerkin matrices
`A_{i+1} = R_i A_i R_i^T` chained from the fine matrix.

Two aggregation strategies tried on the level-1 -> top chain:

* **Principal-axis bucketing** (loop 8): sort centroids along the
  longest spatial extent, slice into K equal buckets per level.
  4:1 coarsening per level.
* **Heavy-edge matching (HEM)** (loop 100, Karypis-Kumar 1998):
  walk node ids in order, pair each unmatched node with its first
  unmatched neighbor, leave the rest as singletons. Aggregates are
  always-connected by construction. 2:1 coarsening per pass.

| problem  | strategy            | hierarchy                                   | result                                          |
|----------|---------------------|---------------------------------------------|-------------------------------------------------|
| 5k       | principal-axis      | 5485 → 23 → 5                               | converged 169 iters / 2.4 s                     |
| 5k       | HEM                 | 5485 → 23 → 12                              | converged 168 iters / 2.4 s                     |
| 81k      | principal-axis      | 81613 → 368 → 92 → 23 → 5                   | stalls at residual ~3.7 after iter 19           |
| 81k      | HEM                 | 81613 → 368 → 186 → 94 → 48 → 26 → 13      | stalls at residual ~3.7 after iter 19           |

Loop-100 finding: **aggregation method is not the bottleneck.**
Both strategies plateau at the same residual on 81k despite very
different aggregate topology (HEM aggregates are always connected,
principal-axis are often disconnected). This rules out the
loop-8 hypothesis.

Loop-100/2 ruled out the kernel hypothesis. `diag_70k_cg_baseline`
(plain unpreconditioned CG on the 81k cut-mesh Laplacian) reports:

  max |row_sum| = 1.16e-10        (constant kernel exact)
  diag range    = [8.2e-2, 1.1e+6]  (7 orders of magnitude!)
  b mean        = -1.3e-13           (b in range(A), consistent)
  plain CG: 200,000 iters → ‖r‖² = 3.6e-10  (133 s)

Implication: the matrix is fine. The constant kernel is essentially
exact and `b` is essentially in range, so the system is consistent.
Plain CG converges to ‖r‖ ≈ 1.9e-5 in 200k iters — that is, the
preconditioned multilevel V-cycle (which stalls at L_inf residual
3.7) is **worse than no preconditioner**.

The smoking gun is the diagonal spread: 7 orders of magnitude.
Galerkin coarsening inherits this range. The intermediate-level
Jacobi smoother (`omega · D^{-1} · r`) becomes useless when D has
that much spread — small-diag rows are over-relaxed, large-diag
rows are under-relaxed. Plain CG's diagonal-Jacobi preconditioner
is implicit and at least homogenizes the scaling per iter; the
V-cycle's discrete Jacobi sweeps don't.

Loop-100/2 also added kernel projection (`zero_mean_in_place`) on
the V-cycle correction. Lean spec + 6 native_decide proofs +
5 RC props, mirrored to `src/curvenet/kernel_projection.h`. Did
not break 5k convergence (168 iters preserved) and was harmless
on 81k (still stalls at 3.7). Kept in the tree because it's
load-bearing infrastructure for any future kernel-aware variant.

Next loop: replace Jacobi smoother with one that handles wide
diagonal ranges — symmetric Gauss-Seidel, or symmetric diagonal
scaling D^{-1/2} A D^{-1/2} on every level before iterating.

## Next steps

Real-mesh row is now interactive. Remaining levers, ranked by
expected impact for going from 11 FPS toward 90 FPS:

1. **Tighten convergence**: 11 FPS at n=5485 is dispatch-overhead-
   light, so the cost is in CG iter count. Add iteration-count
   instrumentation to confirm.
2. **Intrinsic Delaunay flipping** (Sharp & Crane §5) on top of
   mollification — guarantees nonneg edge weights and reduces κ
   further. Bigger code change but the canonical fix.
3. **Incomplete Cholesky preconditioner** on `LhsM_csr` to cut iter
   count by 5-10×.
4. **GPU compute path** finally has a load-bearing CPU baseline to
   measure against. See [PERF_GPU.md](PERF_GPU.md).
5. **fTetWild manifold prepass** per
   `todos/06_ftetwild_runtime_integration.md` for meshes that fail
   the manifold-edges precondition outright.

## When to re-run

Re-run `make -C tests bench` after any of:

* changes to `cut_mesh_laplacian::assemble_lh_csr` /
  `assemble_vt_lh_v_csr`
* changes to `sparse::cg` (preconditioner, tolerance, warm-start)
* changes to the per-frame solve in `curvenet_deformer_3d.cpp`
* swapping the polygon Laplacian implementation
