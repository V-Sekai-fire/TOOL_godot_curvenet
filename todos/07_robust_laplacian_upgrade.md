# DeGoes22 #2 — Upgrade `PolygonLaplacian` to a robust formulation

**Layer:** Lean (`lean/Curvenet/PolygonLaplacian.lean`)
**Possibly C++ runtime mirror once the algorithm lands.**

## Why

The current slice-2 Laplacian fan-triangulates each polygon and sums the
cotangent contribution per triangle. Cotangents blow up on:

* **Near-degenerate triangles.** A triangle with one vertex angle ≈ 0 has a
  near-zero opposing edge cross-product, so `cot(θ) = dot/|cross|` diverges.
  A real character mesh from fTetWild (or any imperfect input) routinely has
  slivers like this — the resulting Laplacian gets diagonal entries on the
  order of 1e10 and the linear solve in slice 5 will be numerically poisoned.
* **Non-manifold inputs.** Even after fTetWild, a shared edge with three
  incident faces (which the cut-mesh can produce around T-junctions) makes
  the cotangent-sum-over-incident-faces ill-defined.
* **Zero-area face after cutting.** §4.1 of DeGoes22 explicitly says cut-faces
  can be non-planar and non-convex; collinear cut-vertices give zero-area
  sub-triangles and divide-by-zero cotangents.

## Candidate references

* **Sharp & Crane 2020, "A Laplacian for Nonmanifold Triangle Meshes"** —
  cleanest replacement for cotangent on triangle inputs. Builds an
  intrinsic Laplacian per non-manifold edge that stays PSD and finite.
* **Bunge, Herholz, Kazhdan, Botsch 2020, "Polygon Laplacian Made Simple"** —
  extends de Goes 2020's polygon Laplacian with a robust virtual-vertex
  construction that avoids the planarity assumption.
* **de Goes, Butts, Desbrun 2020, "Discrete Differential Operators on
  Polygonal Meshes"** — the construction DeGoes22 actually cites; we should
  match it eventually for fidelity to the paper.
* **Stein, Jacobson, Wardetzky, Grinspun 2018, "Natural Boundary Conditions
  for Smoothing in Geometry Processing"** — useful in a different axis
  (boundary handling) but worth revisiting if cut-faces meeting the mesh
  boundary cause artefacts.

## Acceptance

1. Replace `polygonCotLaplacian` body (or add a parallel
   `polygonRobustLaplacian`) using one of the references above. Keep the
   existing function around as a baseline, gated on a tag, so we can A/B
   on instance theorems.
2. New `native_decide` checks: a near-degenerate triangle (one vertex offset
   by 1e-6 along an edge) gives a finite, PSD Laplacian — current code
   produces inf/nan there.
3. Document the chosen formulation in `lean/Curvenet/PolygonLaplacian.lean`'s
   docstring (which currently flags this as a known simplification).

## Priority

After slice 5 lands. Until then the simple cotangent-fan Laplacian is fine
for the small instance theorems we run via `native_decide`; it only becomes
load-bearing once we actually solve on real cut-meshes.
