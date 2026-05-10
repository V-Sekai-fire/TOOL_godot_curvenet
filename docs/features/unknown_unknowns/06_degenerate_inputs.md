# 06 — Numerical robustness on degenerate inputs

## What we don't know

The Sharp & Crane 2020 mollified Laplacian
([05 cot-Laplacian](../known_knowns/05_cot_laplacian.md)) handles thin
triangles. We don't know what happens for:

- Truly zero-length edges (duplicate vertices)
- Triangles with vertices that are exactly collinear
- Curves with zero-length segments (knot duplicates)
- Curves passing through the same vertex twice
- Very small meshes (e.g. ~10 vertices)
- Very large meshes (200 k+ vertices — see also
  [07 scaling](07_scaling_beyond_benchmark.md))
- NaN / Inf coordinates
- Curves longer than the mesh's bounding box

## What might break

- The mollification δ defaults to `1e-5 × mean_edge_length`. With a
  zero-length edge in the input, mean is biased low; mollification
  may not bridge actual gaps.
- `surface_projection::project_to_vertices` is a brute-force
  closest-point search. With ~10 vertices it's fine; with 1 M it
  becomes a measurable bind-step cost.
- The `curvenet_builder::build` ε-merge uses a fixed 1e-6 in the
  deformer call site. Real meshes vary in unit scale (mm vs m vs
  feet vs centimeter). 1e-6 is correct in meters but micrometers in
  millimeter-scale meshes.
- CG convergence on near-singular `LhsM_csr` matrices is not bounded
  in our code. Iterations capped at `max_iter` and we return whatever
  we have. Silent divergence on degenerate input.
- Float overflow in `mat3_inv` for near-singular frames (very tight
  bends in profile curves) is not guarded.

## How we'd find out

- Property tests with adversarial generators: random tiny
  meshes (3-10 verts), random curve placements, random NaN
  injections. Expect crashes / bad outputs; turn each into a test
  case.
- Run the deformer on hand-crafted pathological fixtures:
  `tests/test_degenerate_inputs.cpp` (new) covering each category
  above.
- Fuzz `apply_deformation` with random Curve3D states.

## Mitigation if it breaks

- Per-input validation pass at bind time: reject NaN / Inf with a
  clear error before they propagate.
- ε-merge tolerance scaled to mesh extent (`bbox_diagonal × 1e-6`)
  rather than absolute.
- Guard `mat3_inv` against `|det| < eps`; return identity + a warning
  log.
- Document the supported input domain (mesh size range, scale unit,
  curvature limits) in the artist-facing docs as a known constraint.
