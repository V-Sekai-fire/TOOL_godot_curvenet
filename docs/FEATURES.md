# Features — current state and known gaps

Inventory of what `TOOL_godot_curvenet` ships today and what is still
missing for a production character-articulation pipeline. Snapshot
date: 2026-05-10. See `PERF_BASELINE.md` for measurements,
`IMPOSSIBILITY.md` for the iterative-runtime impossibility result, and
the [archive repo](https://github.com/V-Sekai-fire/TOOL_godot_curvenet_archive)
for retired algorithm families.

## Architecture overview

```
                 +----------------+         +------------------+
artist authoring | Profile curves | -----> | bind step (rare) | --+
 (Curve3D nodes) +----------------+         +------------------+   |
                                                                   v
                                                    +------------------------+
                                                    | rest_cache              |
                                                    |  - cut-mesh (sample-    |
                                                    |    promoted vertices)  |
                                                    |  - L_h CSR, V^T L_h V  |
                                                    |  - DDM influence rows  |
                                                    |  - rest knot positions,|
                                                    |    tilts, widths       |
                                                    +------------------------+
                                                                   |
                                                                   v
artist drag      +----------------+         +------------------+   |
 (knot/tilt/etc) | Handle deltas  | -----> | runtime path     | <-+
                 +----------------+         +------------------+
                                                    |
                          +--- (DDM on)  ---+       +--- (DDM off) ---+
                          v                 v
                +---------+               +-------------------------+
                | LBS-style matvec        | DeGoes22 §6: two-stage  |
                | per vertex over sparse  | PCG (Eq 6a + 6b) on the  |
                | influences (~50 µs/70k) | nv x nv VtLhV system    |
                +---------+               +-------------------------+
```

Two runtime paths:

- **DDM (default-off, opt-in for Quest 3)** — bind-time precompute,
  runtime LBS-flavored matvec, per-handle 4×4 transform built from
  DeGoes22 §3 scaled-frame ratios (translation + tangent rotation +
  tilt + perp-scale width).
- **Direct §6 solve (default-on)** — per-frame two-stage harmonic + Poisson
  PCG against the cached `V^T L_h V` matrix. Warm-started from the
  previous frame's iterate. ICC(0) preconditioner is opt-in via
  `use_incomplete_cholesky`.

Both paths share the bind-time cut-mesh + Laplacian assembly.

## Lean spec layer

**23 modules**, **258 `native_decide` / `decide` proofs** as of commit
`9a8ee1f`. Build: `cd lean && lake build` → 26 jobs, zero errors. Each
module mirrors a C++ header in `src/curvenet/` (or, for DDM, the
`direct_delta_mush.h` C++ header mirrors the Lean module).

| Module | Concept | Proofs |
|---|---|---|
| `Vec3` | 3-vector type | 1 |
| `Halfedge` | halfedge mesh structure + manifold checks | 9 |
| `HalfedgeBuilder` | triangle-soup → HalfedgeMesh | 14 |
| `CurvenetBuilder` | ε-merge knots + classify (anchor / regular / intersection) | 10 |
| `SurfaceProjection` | knot → closest mesh vertex | 10 |
| `PolygonLaplacian` | cot-Laplacian via fan triangulation | 7 |
| `RobustLaplacian` | Sharp-Crane 2020 mollified cot-Laplacian | 14 |
| `CutMesh` | halfedge mesh + sample-promoted vertices | 12 |
| `CutAlgorithm` | segment cutting primitives | 45 |
| `CutMeshLaplacian` | V^T L_h V assembly + null-space + symmetry | 12 |
| `ScaledFrames` | DeGoes22 §3 (BS) and (BS)·(B̆S̆)⁻¹ | 19 |
| `IntersectionFrames` | per-side scaled frames at intersections | 14 |
| `CurveInterp` | per-side normal / width interpolation | 5 |
| `SegmentGradient` | isolated vs intersection gradient dispatcher | 4 |
| `HarmonicSolve` | DeGoes22 §6 Eq. (6a) | 5 |
| `DeformSolve` | DeGoes22 §6 Eq. (6a) + (6b) | 7 |
| `DenseLinAlg` | LU solve, dense matMul | 18 |
| `SparseLinAlg` | CSR + CG with Jacobi precond | 9 |
| `IncompleteCholesky` | ICC(0) factorisation + PCG | 5 |
| `HierarchicalSparsifyCompensate` | greedy IS + Schur compensation | 12 |
| `GraphColoring` | Welsh-Powell greedy | 8 |
| `DirectDeltaMush` | runtime LBS matvec | 6 |
| `DirectDeltaMushBind` | smoothWeights + sparsifyTopK | 12 |

**Zero `sorry` / `admit`.** Each proof exercises the algorithm on a
concrete small instance — the pattern is "spec + run on a tiny example +
verify the answer matches" rather than fully-quantified theorems.

## C++ runtime modules

19 live headers in `src/curvenet/`, all with Lean mirrors.

**Mesh / topology**:
- `vec3.h`, `halfedge.h`, `halfedge_builder.h` — mesh basics
- `curvenet_builder.h` — curve graph (knots + classification)
- `cut_mesh.h`, `cut_algorithm.h` — cut-mesh with sample promotion
- `surface_projection.h` — knot → closest vertex (vertex-only)

**Discretization**:
- `polygon_laplacian.h`, `robust_laplacian.h` — cot weights with
  Sharp-Crane 2020 mollification
- `cut_mesh_laplacian.h` — V^T L_h V assembly (sparse CSR for prod)

**Geometry**:
- `scaled_frames.h` — DeGoes22 §3 deformation gradient
- `intersection_frames.h` — per-side frames at intersections
- `curve_interp.h` — per-side interpolation along curves
- `segment_gradient.h` — isolated / intersection dispatcher

**Solvers**:
- `dense_linalg.h` — small dense LU
- `sparse_linalg.h` — CSR + CG with Jacobi
- `incomplete_cholesky.h` — ICC(0) + PCG
- `hierarchical_sparsify.h` — HSC (used at bind time, opt-in)

**Pipeline**:
- `harmonic_solve.h` — Eq. (6a)
- `deform_solve.h` — Eq. (6a) + (6b)
- `direct_delta_mush.h` — DDM kernel + bind post-processing

**Banned**: Eigen, SuiteSparse, Mathlib (any third-party linear-algebra
or theorem library). All in-house. See `feedback_no_eigen.md` in the
auto-memory.

## RapidCheck test suite

92 RC properties across 13 test binaries, all green.
`make -C tests test` runs them. Each test mirrors a Lean module's
proofs and adds randomized property tests on the C++ implementation.

## Authoring UX (DeGoes22 §3 viewport)

Per-knot 3D-draggable handles in the editor:

| Knob | Handle | Implementation | Visualization |
|---|---|---|---|
| Position | primary dot | snaps to nearest mesh vertex | colored marker by `KnotKind` |
| Tangent in/out | secondary dot (per side) | free-floating in 3D | yellow tangent ray |
| Tilt | secondary dot on rotated b-axis | rotates `Curve3D::point_tilt` | red/green/blue t/n/b axes |
| Width w | secondary dot opposite side of b-axis | sets `knot_widths[curve][knot]` | cyan octagonal ring |

All four undo/redo via `EditorUndoRedoManager`. GDScript-scriptable
through:
- `set_profile_curves`, `get_profile_curves` (existing)
- `set_knot_widths`, `get_knot_widths` — per-curve `PackedFloat32Array`
- `set_knot_width(curve_id, knot_idx, w)`, `get_knot_width(...)` —
  granular accessors used by the gizmo's drag handles
- `set_use_direct_delta_mush(bool)` — toggle DDM runtime path
- `set_ddm_top_k(int)`, `set_ddm_smooth_iters(int)` — DDM bind tuning

## Performance characteristics

See `PERF_BASELINE.md` for full numbers. Headline figures from the
Mire body benchmarks (M2 Pro):

- 5k mesh §6 solve (default): ~3-7 ms / 12 RHS
- 81k mesh §6 solve: ~50-100 ms / 12 RHS (ICC opt-in helps ~2×)
- 5k mesh DDM runtime: < 1 ms (sparse matvec)
- 50k mesh DDM target on Quest 3: ~0.8 ms (planned, not yet measured —
  CPU path is ready, GPU dispatch is the gap)

Bind step cost (one-time per topology change):
- 5k mesh: < 100 ms
- 81k mesh: ~30 s (the dense `harmonic_solve::solve_multi` for DDM
  weight harvest is the dominant cost; sparse multi-RHS PCG harvest
  is the planned optimization)

## Known gaps — features we still need

Ranked by visible impact for character-articulation use cases.

### High-priority

1. **Edge + face surface projection** (`surface_projection.h` is
   vertex-only). Curves can only land on existing mesh vertices →
   visible "stair-step" on coarse meshes. Edge-midpoint and
   face-interior projection would let knots sit anywhere on the
   surface. Spec at the same place in the Lean module would generalize
   `ProjectionKind` enum from `vertex` to also produce
   `edge_intersection` and `face_interior`.
2. **Curve-segment tracing through faces** — once edge/face projection
   lands, the cut-mesh needs to actually cut along curve segments
   (insert cracks into halfedges). `CutAlgorithm.lean` already has the
   surgical primitives (`subdivideEdge`, `splitFace`,
   `insertCrackAtEndpoint`) with 45 native_decide proofs; what's
   missing is the *tracer* that takes a list of projected knots and
   issues the cuts. Without this the deformer treats every curve as a
   sequence of disconnected vertex-snaps.
3. **Quest 3 GPU dispatch for DDM matvec**. Current runtime is CPU.
   Memory-bandwidth-bound: 50k verts × 8 influences × 4 bytes weight ×
   per-frame transforms ≈ 1.6 MB read traffic per frame at 90 Hz — well
   inside Quest 3's bandwidth envelope but only with a Vulkan compute
   dispatch. Pattern already established by `shaders/sgs_color.comp`.
4. **DDM bind harvest on the sparse path** — current bind uses the
   *dense* `harmonic_solve::solve_multi` which is O(nv³). At 81k that's
   ~30 s. Switching to sparse multi-RHS PCG (one CG per handle, warm-
   started against the previous handle's solution) would drop bind to
   ~1 s on 81k.

### Medium-priority

5. **Per-knot side toggle** for asymmetric pull at intersection knots.
   The algorithm side is in `IntersectionFrames` (`per_side_scaled_frames`
   already produces both `+` and `−` sides), but no UX surface — the
   deformer treats curves as if they always pull from the right.
6. **Per-knot length scale `l` drag** — currently auto-derived from
   segment length. DeGoes22 lets the artist override `l` independently
   of `w`. Less commonly authored than `w` but part of the full §3 UX.
7. **Surface-normal-driven reference frame for tilt**. Today the
   tilt-rotation reference is world-Y or world-Z (Gram-Schmidt). On a
   curved surface this means tilt=0 means *aligned with world axes*,
   not *aligned with surface normal*. For DeGoes22-faithful behavior
   the rest reference normal should be the mesh vertex normal at the
   projection point.
8. **Bind cache persistence**. Currently re-binds every Godot session
   on first `apply_deformation`. Saving the cache to a `.cache.bin`
   alongside the scene would skip the bind step on subsequent loads.

### Low-priority / research

9. **Auto profile-curve extraction** from arbitrary meshes —
   skeleton-based (Tagliasacchi 2012 / Coverage Axis 2022) or geodesic
   level sets via Sharp-Crane heat method. Would let the deformer be
   used demo-style on any mesh without artist authoring. Tradeoff:
   curve quality is lower than artist-placed, so deformation quality
   degrades. See README of the archive repo for context.
10. **Heat method on polygon soups** for the auto-extraction route.
    Leverages the in-house `RobustLaplacian` + `SparseLinAlg` +
    `IncompleteCholesky`. Roughly 200 LOC port of
    `nmwsharp/geometry-central`'s heat method to in-house solvers; no
    new dependencies.
11. **Multi-character / batched processing**. Each `CurveNetDeformer3D`
    holds its own `RestCache`. Production rigs with 50+ characters
    would benefit from a shared meshlet / GPU-resident pool.
12. **Animated knot drag** — record-and-replay of knot/tilt/width
    edits as a Godot animation track. Currently authoring is
    real-time-only.

## Out-of-scope / explicitly not pursued

- **Iterative-runtime preconditioners** (multilevel Schwarz, HEM,
  Chebyshev, kernel projection). Ruled out across ~100 loops; see
  `IMPOSSIBILITY.md` and the
  [archive repo](https://github.com/V-Sekai-fire/TOOL_godot_curvenet_archive).
- **Eigen / SuiteSparse / third-party linear algebra**. Banned
  project-wide.
- **Direct Cholesky** at runtime. Would need SuiteSparse / similar →
  banned. ICC(0) is the closest in-house equivalent and is opt-in.
- **Skinning-Eigenmodes-style precompute basis** (Pan 2023 family).
  DDM is the chosen bake-and-stream architecture; SE is a peer with
  different tradeoffs but adopting it would require porting their
  basis-extraction pipeline (substantial work, no clear win over DDM).

## Verification commands

```sh
# Lean spec layer:
cd lean && lake build                # → 26 jobs, zero errors

# C++ unit / property tests:
make -C tests test                   # → 92 RC properties green

# GDExtension build:
scons -j8                            # → framework built, no warnings

# Proof tally:
cd lean/Curvenet && total=0 && \
  for f in *.lean; do \
    total=$((total + $(grep -cE 'native_decide|by decide' "$f"))); \
  done && echo "Total proofs: $total"  # → 258
```

## See also

- `PERF_BASELINE.md` — measurements + the trajectory of ~100 solver
  experiments
- `IMPOSSIBILITY.md` — why iterative-runtime can't hit the Quest 3
  budget; how DDM escapes it
- `PERF_GPU.md` — GPU compute notes (HSC SGS shader; DDM compute is
  the next port)
- `lean/Curvenet/` — every algorithm with native_decide proofs
- `src/curvenet/` — every algorithm in C++ matching the Lean specs
