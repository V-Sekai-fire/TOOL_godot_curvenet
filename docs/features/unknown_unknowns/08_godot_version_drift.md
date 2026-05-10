# 08 — Godot version drift

## What we don't know

`godot-cpp` is pinned to whichever version is checked into
`thirdparty/`. Behavior on older Godot versions (4.5, 4.4) and
future versions (5.0 alpha, beyond) is untested. The GDExtension API
surface and the editor plugin / gizmo APIs in particular have moved
between minor versions before.

## What might break

- `EditorNode3DGizmoPlugin` API has shifted across Godot 4.x. The
  `_redraw`, `_get_handle_value`, `_set_handle`, `_commit_handle`
  signatures have all moved at least once.
- `EditorUndoRedoManager::add_do_method` / `add_undo_method` accept
  variadic args via `Variant` packing. Compile-time and runtime
  argument mismatches surface differently across versions.
- `Curve3D::set_point_tilt` and friends — used in the gizmo's tilt
  drag handle — have been stable but not version-tested.
- `TypedArray<...>` serialization with nested
  `PackedFloat32Array` was added in 4.6; older Godot may treat it as
  a plain Array, breaking the `knot_widths` property.
- Future Godot 5.0 changes are unknown — may introduce breaking
  changes in `GDCLASS` macro semantics, scene file format, or
  `MeshInstance3D` API.
- Runtime / editor split — `register_types` differentiates editor vs
  runtime via macros; misuse on a future Godot may fail to register
  the gizmo in the editor.

## How we'd find out

- CI matrix on Godot 4.5, 4.6, 5.0-alpha (when available).
- Build the GDExtension against multiple `godot-cpp` checkouts (one
  per supported Godot version).
- Smoke test in each Godot version: load the demo scene, verify the
  gizmo appears, drag a knot, verify deformation.

## Mitigation if it breaks

- Pin a specific `godot-cpp` commit in `thirdparty/` and document
  the supported Godot version range.
- `#if GODOT_VERSION_*` guards for any API differences (avoid where
  possible).
- A version-fence test in `tests/` that calls a known-stable godot-cpp
  entry point and confirms expected behavior — fails loudly on drift.
- Document the supported Godot range in the project README.
