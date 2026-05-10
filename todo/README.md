# todo/

Per-feature execution slices for the gaps catalogued in
[`docs/FEATURES.md`](../docs/FEATURES.md). Each todo captures the
**Gall-minimum** next step: a simple working extension of the current
working system, not a full design for the end state. Pick one off the
list, work through it independently, ship it.

## Status legend

- `[ ]` — open, ready to start
- `[~]` — in progress (claim by editing this README)
- `[x]` — landed
- `[-]` — cancelled / superseded (note the reason in the file)

## Index

| #  | Status | Title                                     | Tier   | One-line summary                                                  |
|----|--------|-------------------------------------------|--------|-------------------------------------------------------------------|
| 01 | `[ ]`  | [Edge+face surface projection](01_edge_face_projection.md)             | High   | Knots can land on edges/faces, not just vertices                  |
| 02 | `[ ]`  | [Curve-segment tracing through faces](02_curve_segment_tracing.md)     | High   | Curves cut the mesh, become real surface cracks                   |
| 03 | `[ ]`  | [Quest 3 GPU dispatch](03_quest3_gpu_dispatch.md)                      | High   | DDM runtime matvec on Vulkan compute (CPU stays as fallback)      |
| 04 | `[ ]`  | [Sparse bind harvest](04_sparse_bind_harvest.md)                       | High   | DDM bind step: dense → sparse PCG, 30 s → 1 s on 81 k             |
| 05 | `[ ]`  | [Side toggle UI](05_side_toggle_ui.md)                                 | Medium | Per-knot +/− side toggle for asymmetric pull at intersections     |
| 06 | `[ ]`  | [Per-knot l drag](06_per_knot_l_drag.md)                               | Medium | Third drag-handle kind for length scale (joins tilt + width)      |
| 07 | `[ ]`  | [Surface-normal reference frame](07_surface_normal_reference_frame.md) | Medium | tilt = 0 means "aligned with surface", not "aligned with world Y" |
| 08 | `[ ]`  | [Bind cache persistence](08_bind_cache_persistence.md)                 | Medium | Save / load `RestCache` so re-opening a scene skips the bind step |
| 09 | `[ ]`  | [Auto profile-curve extraction](09_auto_curve_extraction.md)           | Low    | Demo-grade curves from arbitrary meshes (heat-method-driven)      |
| 10 | `[ ]`  | [Heat method on polygon soups](10_heat_method_polygon_soups.md)        | Low    | Robust geodesic distance via Sharp-Crane mollification            |
| 11 | `[ ]`  | [Multi-character batching](11_multi_character_batching.md)             | Low    | Shared meshlet pool for 50+ characters per scene                  |
| 12 | `[ ]`  | [Animated knot drag](12_animated_drag.md)                              | Low    | Record-and-replay of knot edits via Godot AnimationPlayer          |

## Dependency graph

```
01 ──► 02 ──► (05 needs intersection cracks → 02)
04                                            (independent)
03                                            (independent)
07                                            (independent)
06                                            (independent)
08                                            (independent)
10 ──► 09
11 (benefits from 04, but doesn't strictly require it)
12                                            (independent)
```

`01` blocks `02`. `10` blocks `09`. Everything else can start in any order.

## Picking the next todo

When in doubt, work in this order:

1. **04 sparse bind harvest** — smallest, lowest-risk, removes the
   biggest interactive-iteration pain (30 s → 1 s on 81 k).
2. **07 surface-normal reference frame** — second smallest, fixes a
   real tilt semantics bug.
3. **01 edge+face projection** — first new shipping capability.
4. **02 curve tracing** — built on 01.
5. Everything else by tier.

For Quest 3 ship specifically: **03 GPU dispatch** is the only
gating gap; everything else is incremental quality.

## Notes

- Each file is the source-of-truth for that gap's plan. Edit it as you
  learn more during execution.
- If a todo turns out to be wrong (e.g. a measurement falsifies an
  assumption), mark it `[-]` cancelled with a one-line reason — don't
  delete it. Trajectory record matters; see the
  [archive repo](https://github.com/V-Sekai-fire/TOOL_godot_curvenet_archive)
  for the same discipline applied at the algorithm level.
- `docs/FEATURES.md` is the inventory layer; `todo/` is the execution
  layer. The two can drift; that's fine. When something lands, update
  both.
