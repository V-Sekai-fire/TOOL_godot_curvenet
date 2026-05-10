# 10 — Heat method on polygon soups

## Why (current pain)

Auto-curve-extraction (todo 09) and any future surface-aware tool
needs robust geodesic distance on arbitrary input meshes — including
non-manifold / broken meshes that production pipelines actually
produce. Crane 2013's heat method gives this in three CG solves; the
Sharp & Crane 2020 mollified Laplacian (already in the codebase as
`RobustLaplacian`) makes it work on polygon soups.
Low/research priority gap in
[`docs/FEATURES.md`](../docs/FEATURES.md#low-priority--research).

## Gall-minimum slice

- **In scope**: ~200 LOC port of `nmwsharp/geometry-central`'s heat
  method on top of the in-house solvers. Three solves per query:
  `(M − tL)·u = δ_source`, normalize `−∇u`, then `L·φ = ∇·X`.
- **Deferred**: signed distance fields (Feng 2024 generalization);
  intrinsic Delaunay refinement (extra mile beyond what mollification
  alone provides).
- **Why this slice**: gives geodesic distance at the algorithm tier
  without adding dependencies. Once it's there, todo 09 (and any
  future surface tool) gets it for free.

## Files to touch

- `lean/Curvenet/HeatMethod.lean` (new) — spec the three-step solve
  on a 2-tri strip with native_decide checks (source vertex distance
  = 0, neighbor distance ≈ edge length).
- `src/curvenet/heat_method.h` (new) — C++ mirror. Public API:
  `geodesic_distance(mesh, positions, source_vertex) → vector<double>`.
- `tests/test_heat_method.cpp` (new) — RC properties: source
  distance = 0; symmetry on equilateral triangle; CG converges.

## Approach

- Step 1 (heat diffusion): build `M - tL` where `M` is the lumped
  vertex-area mass matrix (diagonal) and `L` is the cot-Laplacian
  from `RobustLaplacian` (mollified, polygon-soup-safe). Set
  `t = h²` where `h` is mean edge length. Right-hand side is a unit
  delta at the source vertex. Solve via
  `incomplete_cholesky::cg_icc_with_guess` if available, else
  `sparse::cg_with_guess`.
- Step 2 (gradient + normalize): on each triangle, compute
  `∇u` via the standard barycentric formula, normalize to unit length,
  negate. This is the unit-direction field pointing away from the
  source.
- Step 3 (Poisson recovery): build per-vertex divergence
  `div(X)[v] = sum over incident triangles of weighted edge dot X`.
  Solve `L·φ = div(X)` via the same CG path. Result is the
  geodesic distance field up to a constant — shift so source = 0.
- The full implementation reuses every primitive we already have;
  no new linear algebra.

## Verification

- **Lean**: `lake build` clean. `Curvenet.HeatMethod` adds ~6 proofs.
- **C++**: ~6 RC properties on small meshes (single triangle with
  source at one corner; 2-tri strip; small grid).
- **GDExtension**: no GDExtension impact (pure algorithm header).
- **Manual**: spot-check on a sphere — geodesic distance from north
  pole should match `r · θ` analytically up to mesh resolution error.

## Blocks / blocked-by

- **Blocks**: 09.
- **Blocked-by**: nothing — `RobustLaplacian` is already proved
  green (14 native_decide), CG and ICC are already in the codebase.

## Estimated cost

- LOC: ~250 across Lean (~80) + C++ heat_method.h (~150) + tests
  (~70).
- Effort: medium.
- Risk: low. The reference implementation in geometry-central is
  battle-tested; we're porting it to in-house solvers, not inventing.
