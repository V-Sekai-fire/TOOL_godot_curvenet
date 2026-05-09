# TOOL_godot_curvenet

A Godot 4 GDExtension implementing Pixar-style **CurveNet** profile-curve articulation
for triangle meshes, with on-the-fly tris-to-quads conversion so the Coons patches
have well-defined four-sided domains.

## Why

Pixar's CurveNet (SIGGRAPH 2023, *Character Articulation through Profile Curves*)
deforms a surface by interpolating between **closed Bezier profile curves** that ride
on the mesh. Each face becomes a Coons patch whose four boundary curves are segments
of the surrounding profile curves; moving a curve handle deforms the surface in a
way that respects silhouettes — a property cage-based deformers don't have.

CurveNet wants quad faces. Godot is a triangle-primary engine. So this extension
runs a **maximum-weight matching pass on the input triangle mesh** to fuse pairs of
triangles into quads at runtime (the same trick as Blender's Tris-to-Quads, but as
exact graph matching via LEMON's Edmonds-blossom rather than ILP).

## Architecture

```
src/curvenet/                Pure C++ math, no godot-cpp dependency.
  vec3.h                     3D vector.
  bezier.h                   Cubic Bernstein evaluator + derivative.
  profile_curve.h            Closed loop of cubic Beziers, shared endpoints,
                             per-handle in/out tangents (creases supported).
  coons_patch.h              Bilinear-blended Coons patch over 4 boundary curves.
  ngon_patch.h               N-sided patch (N=4 -> Coons; N!=4 TODO mean-value).
  tris_to_quads.{h,cpp}      Triangle-pair fusion via LEMON max-weight matching.
src/                         GDExtension binding layer (register_types.cpp).
tests/                       RapidCheck property tests; godot-cpp-free.
thirdparty/
  godot-cpp/                 Godot 4.5-stable bindings.
  rapidcheck/                Property-based testing (BSD).
  lemon/                     Library for Efficient Modeling and Optimization in
                             Networks (Boost license). Used for Edmonds blossom
                             max-weight matching.
```

## Build

```sh
scons                          # builds the GDExtension via godot-cpp
make -C tests test             # runs RapidCheck property tests (godot-cpp-free)
cd lean && lake build          # checks the Lean4 formalization
```

The Makefile under `tests/` builds independently of godot-cpp so the math layer
iterates without a Godot toolchain. The Lean4 project under `lean/` mirrors
the math as `def`s and `native_decide`s concrete property checks.

## Status (TDD progress)

| Cycle | What | Properties | Status |
|-------|------|------------|--------|
| 1 | Cubic Bezier eval + derivative | 6 | green |
| 2 | Closed-loop profile curve | 7 | green |
| 3 | Bilinear Coons patch (quads) | 6 | green |
| 4 | N-gon patch (N=3, 4, ≥5 via MVC) | 8 | green |
| 5 | Mesh binding & deformation | 5 | green |
| 6 | `CurveNetDeformer3D` Godot node | — | full pipeline live + cached + GDScript API |
| 7 | Tris-to-quads via LEMON matching | 5 | green |
| 8 | LEMON `-fno-exceptions` patch | — | green |
| 9 | Lean4 proof companion | 24 theorems | green (`lake build` passes) |
| — | Bilinear inverse (Gauss-Newton) | 7 | green |
| — | Mesh binding (`bind_polymesh`) | 4 | green |
| — | Mean-value coords (Floater 2003) | 4 | green |
| — | End-to-end pipeline integration | 3 | green |

**RapidCheck:** 55 properties × 100 random cases = **5,500 checks passing**.
**Lean4:** 24 `native_decide` theorems (Bezier, Coons, NgonPatch tri-path,
mean-value Lagrange + partition of unity). Generic-over-ℝ theorems require
Mathlib (deferred).

## GDScript API

`CurveNetDeformer3D` extends `MeshInstance3D` and exposes:

```gdscript
var d := CurveNetDeformer3D.new()
d.source_path = NodePath("../SourceMesh")
d.profile_curves = [my_curve_3d]
d.length_tiebreak = 0.1            # 0..1, weights longer dissolved edges higher
d.deformation_active = true        # triggers apply_deformation()

d.apply_deformation()              # rest-pose pipeline cached after first call
d.get_face_count()                 # post tri->quad fusion
d.get_face_vertex_count(i)         # 3 (triangle) or 4 (quad)
d.evaluate_face(i, s, t)           # Vector3 patch evaluation at (s, t) ∈ [0,1]²
```

## Acknowledgements

- Pixar's *Character Articulation through Profile Curves*, SIGGRAPH 2023.
- Vendored: Godot godot-cpp, RapidCheck, LEMON.
- The existing biharmonic cage deformer
  (`V-Sekai-fire/TOOL_godot_cage_deformer`) provided the project skeleton and
  the `VertexHandles` editor gizmo addon, which cycle 6 will reuse for
  authoring profile-curve control points.
