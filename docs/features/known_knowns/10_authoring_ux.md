# 10 — Authoring UX

## What it is

Four 3D-draggable handles per profile-curve knot, covering the full
DeGoes22 §3 frame: position, tangent in/out, tilt rotation around
tangent, and binormal-direction width. Every handle has full
undo/redo via `EditorUndoRedoManager` and a corresponding GDScript-
scriptable property surface.

## Status

shipping default — the gizmo is registered when the deformer is
selected in the editor. All four knob kinds work today; on a curved
surface the tilt reference is world-Y-up not surface-normal — see
[known unknowns 07](../known_unknowns/07_surface_normal_reference_frame.md).

## Files

### C++ implementation
- `src/curvenet_gizmo_plugin.cpp` — handles definitions, drag math,
  undo registration. Anonymous namespace helpers
  (`enumerate_all_knots`, `compute_knot_frame`,
  `enumerate_tangent_refs`).
- `src/curvenet_gizmo_plugin.h` — plugin declaration
- `src/curvenet_editor_plugin.cpp` — registers the gizmo plugin
- `src/curvenet_deformer_3d.{h,cpp}` — exposes the underlying
  properties to GDScript

### Tests
- No automated UX tests — manual verification in-editor

## API surface

### GDScript / inspector

- `profile_curves : Array<Curve3D>` — the curves
- `knot_widths : Array<PackedFloat32Array>` — per-curve width arrays
  parallel to `point_count`. Defaults to 1.0 when missing.
- `set_knot_width(curve_id, knot_idx, w)` /
  `get_knot_width(curve_id, knot_idx)` — granular accessors used by
  the gizmo's drag handles
- Curve3D's existing per-point `tilt`, `point_in`, `point_out`,
  `point_position` — used directly for tilt + tangent + position

### Gizmo handle id encoding

```
primary:    [0, n_merged_knots)        — knot position drag
secondary:  [0, nT)                    — tangent in/out
            [nT, nT + nK)              — tilt rotation
            [nT + nK, nT + 2*nK)       — width drag
```

`nT` = enumerated tangent refs (in + out per (curve, knot) entry in
the merged graph), `nK` = total knot count across all curves.

## How it works

- **Position drag** (primary): raycast hit projected to a camera-
  facing plane through the merged-knot anchor; snap to closest
  source-mesh vertex; propagate the new position to every Curve3D's
  knot in the incidence list (so merged knots stay merged).
- **Tangent in/out** (secondary, [0, nT)): free-floating drag in 3D;
  no surface snap.
- **Tilt** (secondary, [nT, nT+nK)): drag handle at
  `origin + b · TILT_HANDLE_RADIUS` in the rotated frame. Hit
  projects to plane perpendicular to tangent through origin; new
  tilt = `atan2(proj·n0, proj·b0)` against the world-up Gram-
  Schmidt reference.
- **Width** (secondary, [nT+nK, nT+2*nK)): drag handle on the
  opposite side of the rotated b-axis at distance
  `WIDTH_HANDLE_RADIUS · w`. Hit projects to same perpendicular
  plane; new w = `|proj| / WIDTH_HANDLE_RADIUS`, clamped to
  [0.05, 50.0].
- All four use `EditorUndoRedoManager::create_action` /
  `add_do_method` / `add_undo_method` so the standard editor undo
  stack covers them. Cancel mid-drag rolls back via `p_restore`.

## Cross-references

- Visualization in [11 editor visualization](11_editor_visualization.md)
- Runtime path: knobs feed `F_h` in [09 DDM](09_direct_delta_mush.md)
  or `F_v`/`X_v` in [08 §6 solve](08_degoes22_solve.md)
- Missing knobs ranked in
  [known unknowns 05 (side toggle)](../known_unknowns/05_side_toggle_ui.md),
  [06 (l drag)](../known_unknowns/06_per_knot_l_drag.md)
- Surface-normal correction:
  [07](../known_unknowns/07_surface_normal_reference_frame.md)
