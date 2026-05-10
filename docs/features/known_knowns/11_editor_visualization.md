# 11 — Editor visualization

## What it is

Viewport rendering layer drawn by `CurveNetGizmoPlugin::_redraw`. Five
visual cues that make curvenet structure legible in the Godot 3D
editor:

1. **Knot kind markers** — small "+"-cross at each merged knot, color-
   coded by classification (anchor red, regular white, intersection
   green).
2. **Tangent rays** — yellow short rays at intersection knots showing
   the outgoing tangents.
3. **Tangent control links** — yellow lines from each Curve3D anchor
   to its `point_in` / `point_out` handle.
4. **Per-knot frame axes** — tangent (red), tilt-rotated normal
   (green), tilt-rotated binormal (blue). Makes tilt visible.
5. **Width ring** — cyan octagon perpendicular to tangent, radius
   scaled by per-knot width. Makes width visible.
6. **Projection links** — cyan lines from each merged knot to its
   closest mesh vertex (the projection target).

## Status

shipping default — drawn whenever a `CurveNetDeformer3D` is selected
in the 3D editor.

## Files

### C++ implementation
- `src/curvenet_gizmo_plugin.cpp::_redraw` — all visualization code
- `src/curvenet_gizmo_plugin.cpp::_init` — material registration
  (anchor / regular / intersection / tangent / tangent_link /
  projection_link / frame_t / frame_n / frame_b / width_ring /
  knot_handles / tangent_handles / tilt_handles / width_handles)

### Tests
- No automated viz tests — visual correctness verified in-editor

## API surface

None — pure rendering, no GDScript interface.

## How it works

- The redraw rebuilds the curvenet graph from the current
  `profile_curves` (cheap; deterministic insertion order).
- Per-curve, per-knot it computes the local frame via
  `compute_knot_frame` (tangent from neighbor knots, world-up Gram-
  Schmidt for the binormal reference, tilt rotation from
  `Curve3D::get_point_tilt`).
- Lines are accumulated into per-color `PackedVector3Array`s and
  pushed via `gizmo->add_lines` with the matching material.
- Width ring is rendered as 8 short line segments in the (b, n)
  plane, scaled by the per-knot width from `knot_widths`.
- Projection links are recomputed each redraw using
  `surface_projection::project_to_vertices` against the source
  mesh's vertex positions — cheap for typical curvenets, kept
  in-editor for live feedback.

## Cross-references

- Mirrors the runtime frame derivation in
  [09 DDM](09_direct_delta_mush.md) — same world-up Gram-Schmidt
  + tilt rotation pattern
- Drag handles registered alongside in
  [10 authoring UX](10_authoring_ux.md)
- Better tilt + width with surface-normal reference:
  [known unknowns 07](../known_unknowns/07_surface_normal_reference_frame.md)
