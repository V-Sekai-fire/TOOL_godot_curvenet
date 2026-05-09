# TOOL_godot_curvenet

A Godot 4 GDExtension implementing Pixar's **Profile Curves** character
articulation algorithm (de Goes, Sheffler, Fleischer, *SIGGRAPH 2022*) for
triangle meshes — paired with a Lean 4 spec that proves the algorithm's
load-bearing invariants on small instances via `native_decide`.

## Why

DeGoes22 deforms a surface by solving a two-stage Poisson system on a
**cut-mesh** that splits triangles wherever a curvenet of cubic Bézier
curves crosses them. Each curvenet knot carries scaled per-side frames
that drive the deformation gradient field; the resulting interpolation
respects silhouettes and detail in a way cage- and skinning-based
deformers don't. Compared to the original tris-to-quads + Coons-patch
approach this project started with, the DeGoes22 path:

* runs on the input triangle mesh as-is — no quad-fusion pre-pass
* handles intersections, anchors, and shared knots in arbitrary
  curvenet topologies (panda-style multi-curve rigs, not just one
  closed loop)
* propagates deformation globally via a sparse harmonic solve, so a
  single moved knot smoothly affects the whole region

See `bin/DeGoes22.pdf` (and `bin/macos/3587421.3595415.pdf` for the
Elemental23 follow-up) for the full paper text. The Lean spec under
`lean/Curvenet/` mirrors §3 (curvenet representation), §4.1
(cut-mesh), §4.2 (discretisation), and §4.3 (two-stage solve).

## Architecture

```
src/                                     GDExtension binding layer
  curvenet_deformer_3d.{h,cpp}           CurveNetDeformer3D MeshInstance3D
  curvenet_gizmo_plugin.{h,cpp}          Editor gizmo: knot markers,
  curvenet_editor_plugin.{h,cpp}         tangent handles, projection
                                         lines, drag-to-edit + undo
  vertex_handles_3d.{h,cpp}              Per-vertex source-mesh editor
  vertex_handles_gizmo_plugin.{h,cpp}    Wireframe + handles gizmo
  vertex_handles_editor_plugin.{h,cpp}   Plugin registration
  register_types.cpp                     ClassDB hooks

src/curvenet/                            Pure C++ runtime, header-only
  vec3.h                                 3-vector
  halfedge.h, halfedge_builder.h         Halfedge mesh + tri-soup ctor
  cut_mesh.h, cut_algorithm.h            §4.1 cut-mesh + surgery
                                         (subdivideEdge, splitFace,
                                         insertCrack)
  curvenet_builder.h                     ε-merge knots across input
                                         Curve3Ds; classify anchor /
                                         regular / intersection
  surface_projection.h                   Project knots → mesh vertices,
                                         promote to samples
  intersection_frames.h                  §3 corner normals, per-side
                                         widths, scaled frame B·S
  scaled_frames.h                        Mat3 toolbox + smallest
                                         rotation + F = (B S)(B̆ S̆)⁻¹
  curve_interp.h                         Linear blend of (n, w) along
                                         a curve between intersections
  segment_gradient.h                     Per-segment F dispatcher
  polygon_laplacian.h                    Per-face cot Laplacian
                                         (fan-triangulation; robust
                                         upgrade in todos/07)
  cut_mesh_laplacian.h                   Lₕ + V + VᵀLₕV assembly
  dense_linalg.h                         Mat ops, Gauss elim, LU
                                         factor-once / solve-many
  sparse_linalg.h                        CSR + preconditioned CG
                                         (foundation for the GPU
                                         compute path in todos/08)
  harmonic_solve.h                       Eq. (6a) k-column harmonic
  deform_solve.h                         Full §4.3 two-stage pipeline

lean/Curvenet/                           Lean4 spec, ~136 native_decide
                                         proofs over the same algorithm

tests/                                   RapidCheck property tests
thirdparty/
  godot-cpp/                             Godot 4.5-stable bindings
  rapidcheck/                            Property-based testing (BSD)

todos/                                   Deferred-work design notes
  06_ftetwild_runtime_integration.md     Manifold prepass for arbitrary
                                         user-imported geometry
  07_robust_laplacian_upgrade.md         Sharp 2020 / Bunge 2020 /
                                         de Goes 2020 polygon Laplacian
  08_gpu_compute_solver.md               Compute-shader CG for 100k
                                         real-time on Steam Deck +
                                         Quest 3
```

## Build

```sh
scons                          # builds the GDExtension via godot-cpp
make -C tests test             # runs RapidCheck property tests (godot-cpp-free)
cd lean && lake build          # checks the Lean4 formalization
```

The Makefile under `tests/` builds independently of godot-cpp so the math
layer iterates without a Godot toolchain. The Lean4 project under `lean/`
mirrors the math as `def`s and `native_decide`s concrete property checks.

## Pipeline status

The DeGoes22 rewrite is complete in Lean (20 slices) and mirrored 1:1 in
header-only C++. The runtime is wired into `CurveNetDeformer3D` and
produces sane deformation under the §4.3 solve.

| Layer | Lean module | C++ header | Tests |
|------|-------------|-----------|-------|
| §3 curvenet | `IntersectionFrames`, `CurveInterp`, `SegmentGradient`, `ScaledFrames` | matching `.h` | `test_scaled_frames.cpp` |
| §4.1 cut-mesh | `Halfedge`, `CutMesh`, `CutAlgorithm` | matching `.h` + `halfedge_builder.h` | `test_halfedge.cpp`, `test_cut_algorithm.cpp`, `test_halfedge_builder.cpp` |
| §4.2 discretisation | `PolygonLaplacian`, `CutMeshLaplacian` | matching `.h` | `test_polygon_laplacian.cpp` |
| solver kernels | `DenseLinAlg`, `SparseLinAlg` | matching `.h` | `test_dense_linalg.cpp` |
| §4.3 solve | `HarmonicSolve`, `DeformSolve` | matching `.h` | `test_deform_solve.cpp` |
| runtime glue | (none — Godot-side) | `curvenet_builder.h`, `surface_projection.h`, `curvenet_deformer_3d.{h,cpp}` | `test_curvenet_builder.cpp`, `test_surface_projection.cpp` |

**RapidCheck:** 55 properties × 100 random cases = **5,500 runtime checks
passing.**
**Lean4:** ~136 `native_decide` proofs across 14 modules.

## GDScript API

`CurveNetDeformer3D` extends `MeshInstance3D`:

```gdscript
var d := CurveNetDeformer3D.new()
d.source_path = NodePath("../SourceMesh")
d.profile_curves = [my_curve_3d, another_curve_3d]   # ε-merge across all
d.deformation_active = true                          # fires apply_deformation()
d.apply_deformation()                                # rest-pose cache built
                                                     # at first call
```

In the editor, selecting a `CurveNetDeformer3D` activates the
`CurveNetGizmoPlugin`:

* coloured `+` markers per merged curvenet knot
  (red anchor / white regular / green intersection)
* yellow tangent rays at intersections + dashed links to drag-handles
  for `point_in` / `point_out` per Curve3D handle
* cyan lines from each knot to its closest mesh vertex (the runtime
  sample-promotion target)
* drag a knot — every Curve3D handle that ε-merged into it follows in
  lockstep, the result snaps to the nearest source-mesh vertex, and
  the deformer re-solves immediately
* drag a tangent — `set_point_in` / `set_point_out` updates the
  underlying Curve3D, no surface snap (tangents are control points)
* both have undo / redo via `EditorUndoRedoManager`

## Acknowledgements

Bibliographic citations live in `references.bib`. The two load-bearing
papers are:

- `DeGoes2022Curvenet` — de Goes, Sheffler, Fleischer, *Character
  Articulation through Profile Curves*, SIGGRAPH 2022.
- `Nguyen2023Elemental` — Nguyen et al, *Shaping the Elements:
  Curvenet Animation Controls in Pixar's Elemental*, SIGGRAPH 2023
  Talks.

Plus references for deferred-work upgrades (robust polygon Laplacian,
mean-value coords, Polthier–Schmies straightest-path, fTetWild) cited
from the matching `todos/` design notes.

Vendored: Godot godot-cpp, RapidCheck. The existing biharmonic cage
deformer (`V-Sekai-fire/TOOL_godot_cage_deformer`) provided the project
skeleton and the `VertexHandles` editor gizmo addon (subsequently
ported to native C++).
