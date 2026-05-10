# 13 — Godot CSG integration

## Why (current pain)

`CurveNetDeformer3D::source_path` resolves through
`Object::cast_to<MeshInstance3D>` only. CSG nodes (`CSGBox3D`,
`CSGCombiner3D`, `CSGMesh3D`, `CSGShape3D`) generate their geometry
at runtime via boolean operations and aren't subclasses of
`MeshInstance3D` — so there's no path today to drive the deformer
from a CSG-authored mesh. CSG is a standard Godot workflow for
blockout characters, procedural props, and editable boolean rigs;
not supporting it is a real authoring-pipeline gap.

## Gall-minimum slice

- **In scope**: extend the source-node dispatch in the bind step to
  accept any `Node3D` whose runtime output exposes a `Mesh` resource.
  For `CSGShape3D` and subclasses, call `bake_static_mesh()` once at
  bind time and proceed through the existing pipeline (halfedge
  builder → cut-mesh → cot-Laplacian).
- **Deferred**: live CSG-mesh tracking (auto-rebind every time the
  artist moves a CSG operand — would require listening on the CSG
  node's `tree_exiting` / property change signals); deformer output
  → CSG operand wiring (the inverse direction, where the deformed
  mesh becomes a CSG operand for further booleans).
- **Why this slice**: ~80 LOC. Sharp & Crane mollification already
  handles the non-manifold geometry CSG produces, so the math layer
  is unchanged. Bake-once-at-bind matches the existing cache lifecycle
  exactly.

## Files to touch

- `src/curvenet_deformer_3d.cpp` — in `apply_deformation`'s rebuild
  block, replace the `Object::cast_to<MeshInstance3D>` lookup with a
  small dispatch helper that handles `MeshInstance3D` (existing) and
  `CSGShape3D` / `CSGCombiner3D` (new).
- `tests/test_csg_integration.cpp` (new) — smoke test exercising the
  source-node dispatch with a fixture CSGShape3D-backed mesh.

## Approach

- New helper in the deformer's anonymous namespace:
  ```cpp
  Ref<Mesh> get_source_mesh(Node3D *node) {
      if (auto mi = Object::cast_to<MeshInstance3D>(node)) {
          return mi->get_mesh();
      }
      if (auto csg = Object::cast_to<CSGShape3D>(node)) {
          return csg->bake_static_mesh();
      }
      return Ref<Mesh>();
  }
  ```
- The rebuild block calls this helper; everything downstream
  (vertex extraction, triangle indices, halfedge build) is unchanged.
- Bind-cache hash includes the baked mesh's content fingerprint;
  topology changes from CSG operand moves automatically invalidate.
- Document the limitation: bind only fires on `apply_deformation`,
  so artists driving CSG live need to call the deformer's
  `apply_deformation` after each CSG edit (or wire a signal connection
  in a follow-up).

## Verification

- **Lean**: no change. The math layer doesn't care about source
  provenance.
- **C++**: `tests/test_csg_integration.cpp` adds ~3 RC properties:
  - source = `MeshInstance3D` → returns its mesh
  - source = `CSGShape3D` → returns the baked mesh
  - source = neither → returns null Ref (deformer falls through to
    the no-op early-return path)
- **GDExtension**: `scons -j8` clean. No new dependencies — both
  `MeshInstance3D` and `CSGShape3D` are already linked via
  `godot-cpp`.
- **Manual**: drop a `CSGBox3D` into a scene, attach a
  `CurveNetDeformer3D`, set `source_path` to the CSG node, add a
  profile curve, drag a knot. Verify deformation runs and the
  CSG-baked mesh deforms as expected.

## Blocks / blocked-by

- **Blocks**: nothing.
- **Blocked-by**: nothing.
- **Related**: a future "live CSG tracking" follow-up would need the
  bind cache persistence work in
  [08](08_bind_cache_persistence.md) so frequent rebinds don't
  thrash. Auto-rebind on CSG signal is its own todo if appetite
  warrants.

## Estimated cost

- LOC: ~80 across deformer dispatch (~50) + tests (~30).
- Effort: small.
- Risk: low. `CSGShape3D::bake_static_mesh` is stable Godot 4.x API;
  the resulting mesh is just an `ArrayMesh` with the same surface
  arrays the deformer already consumes. Non-manifold tolerance is
  inherited from [05 cot-Laplacian](../known/05_cot_laplacian.md)'s
  Sharp-Crane mollification.
