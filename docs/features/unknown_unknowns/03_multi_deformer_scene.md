# 03 — Multi-deformer scene behavior

## What we don't know

Every test has run with **one** `CurveNetDeformer3D` in the scene.
Behavior with 5 / 10 / 50 deformers at once — separate
`RestCache` per node, parallel `apply_deformation` triggers,
contended editor-undo stack — is untested. Production scenes
(crowds, multi-character tools) need this, but we have no data.

## What might break

- Memory ceiling: each `RestCache` on 81 k holds CSR matrices and
  ICC factor totalling ~50 MB. At 50 deformers that's 2.5 GB just
  for solver state — exceeds typical workstation limits.
- Editor-undo stack: every drag commits actions globally. Multiple
  deformers' undo histories are interleaved on a single stack;
  cross-deformer undo ordering may behave unexpectedly.
- Gizmo registration: `_create_gizmo` connects a redraw signal per
  deformer. Many deformers → many redraws on every editor frame.
  Frame cost untested.
- `_curvenet_redraw_request` is a per-instance signal name, but the
  `Callable` binding pattern hasn't been stress-tested for
  per-instance correctness across many instances.
- Concurrent `apply_deformation` calls from script — each touches its
  own `rest_cache`, but shared resources (the source mesh's vertex
  positions) might race.

## How we'd find out

- Build a stress scene: 20 `CurveNetDeformer3D` nodes, each with a
  small character mesh (~5k each) and 3 curves.
- Profile editor frame time, memory, redraw frequency.
- Drag knots in two different deformers in quick succession, verify
  undo/redo order makes sense.
- Save the scene, close, reopen — does every deformer rebind cleanly?

## Mitigation if it breaks

- [Known unknowns 11 — multi-character batching](../known_unknowns/11_multi_character_batching.md)
  is the planned solution. A shared mesh pool + per-instance state
  lets 50 characters share matrix factors.
- Per-deformer "lazy rebind" — defer the bind until first apply
  rather than on selection.
- Redraw throttling — coalesce redraws across deformers in a single
  editor frame.
