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
| 4 | N-gon patch scaffold (N=4 path) | 3 | green |
| 5 | Mesh binding & deformation | 5 | green |
| 6 | `CurveNetDeformer3D` Godot node | — | scons-builds; deform-from-curves WIP |
| 7 | Tris-to-quads via LEMON matching | 5 | green |
| 8 | LEMON `-fno-exceptions` patch | — | green (extension compiles cleanly) |
| 9 | Lean4 proof companion | 12 instance theorems | green (`lake build` passes) |

**RapidCheck:** 32 properties × 100 random cases = 3,200 checks passing.
**Lean4:** 12 `native_decide` corner-recovery theorems passing on Float-valued
specifications mirroring the C++ math. Generic-over-ℝ theorems deferred until
Mathlib is wired in.

## Acknowledgements

- Pixar's *Character Articulation through Profile Curves*, SIGGRAPH 2023.
- Vendored: Godot godot-cpp, RapidCheck, LEMON.
- The existing biharmonic cage deformer
  (`V-Sekai-fire/TOOL_godot_cage_deformer`) provided the project skeleton and
  the `VertexHandles` editor gizmo addon, which cycle 6 will reuse for
  authoring profile-curve control points.
