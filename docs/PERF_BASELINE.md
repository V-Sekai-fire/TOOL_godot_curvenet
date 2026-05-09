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

* **Solve = 163 sec for 12 RHS at 81k (unchanged).** CG iter count
  grows faster than √n on character-topology cot-Laplacians; per-RHS
  cold-start is 13.6 s. Warm-start halves it but the architecture
  needs to change for the 11 ms / 90 FPS target — meshlet
  decomposition or GPU compute remain the load-bearing options.

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
