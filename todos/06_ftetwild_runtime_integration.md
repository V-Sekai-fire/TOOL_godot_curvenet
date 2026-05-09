# DeGoes22 #1 — fTetWild manifold-repair preprocessor (C++ runtime)

**Layer:** C++ runtime (`src/`), thirdparty integration

## Why

`Curvenet.Halfedge.manifold?` is a hypothesis the Lean side asserts on its
input. The DeGoes22 cut-mesh algorithm (§4.1) breaks if the input surface
isn't an oriented manifold (with boundary OK). To guarantee the hypothesis
at runtime against arbitrary user-imported geometry, we need a manifold
repair pass.

User picked **fTetWild** (https://github.com/wildmeshing/fTetWild).

Caveats:
- fTetWild is a *tetrahedralizer*, not a surface repair tool. To get a manifold
  surface back out we'd run fTetWild then extract the boundary triangles of
  the produced tet mesh.
- Heavy dependencies: Eigen, GeoGram, libigl, CGAL. Vendoring is non-trivial.
- Output topology can differ significantly from input (especially on
  non-manifold inputs the user didn't intend to be repaired).

## Scope

1. Vendor `thirdparty/fTetWild/` (or add as a git submodule).
2. SConstruct integration: build fTetWild + its deps as static libs and link
   into the GDExtension.
3. Wrap fTetWild in a thin C++ helper `curvenet::ensure_manifold(input) -> manifold`:
   - if `is_manifold(input)` already, return as-is (no tet conversion);
   - else run fTetWild and extract surface.
4. Hook into `CurveNetDeformer3D::apply_deformation` *before* cut-mesh
   construction (slice 3 of the Lean rewrite).

## Alternatives considered (rejected, kept for context)

The fTetWild choice is **locked in** (see project memory entry
`manifold_prepass_ftetwild`). The alternatives below were considered but
not selected — listed here only so future readers don't re-relitigate.

- **MeshFix** (https://github.com/MarcoAttene/MeshFix-V2.1): lighter,
  surface-only repair. Less robust on pathological cases.
- **libigl `is_edge_manifold` + bowtie removal**: cheap if input is
  *almost* manifold. Doesn't handle self-intersecting input.
- **Trust the input**: rejected — fine for the demo (Godot primitives are
  manifold) but breaks for arbitrary user-imported assets.

## Acceptance

- `scons` clean build with fTetWild vendored.
- A unit test feeds a non-manifold triangle soup (e.g., a self-intersecting
  bowtie) through `ensure_manifold` and the result satisfies `is_manifold`.
- The deformer pipeline runs `ensure_manifold` once at bind time (cached) so
  per-frame cost is unaffected.
