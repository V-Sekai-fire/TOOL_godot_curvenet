# 04 — Scene save / load lifecycle

## What we don't know

Editor lifecycle around the deformer is untested across scene save,
close, reopen, and reload. We don't know if the property surface
serializes correctly, if cached state (`prev_Fv`, `prev_Xv`,
`ddm_influences`) survives, or if `apply_deformation` on a freshly
loaded scene behaves the same as on a freshly built one.

## What might break

- `RestCache` is `mutable` and built lazily in `apply_deformation`.
  Serialization should NOT include it (it's recoverable). Untested
  whether Godot serializes anything we don't want.
- `knot_widths : TypedArray<PackedFloat32Array>` — Godot's TypedArray
  ↔ scene-file serialization for nested PackedFloat32Array hasn't
  been exercised. If it round-trips lossily, widths reset on reload.
- The bind step does `~30s` on 81k. If reopening a scene triggers
  an automatic `apply_deformation`, the editor freezes. We don't
  cache the bind output ([known unknowns 08](../known_unknowns/08_bind_cache_persistence.md)).
- Curve3D resources are sub-resources of the scene. If two scenes
  share a Curve3D resource (extracted to a `.tres`), edits in one
  affect the other — including the deformer's "rest" snapshot.
  Untested.
- Scene reload (`Ctrl+Shift+T`) re-instantiates the deformer. The
  gizmo's `_create_gizmo` should fire again; signal connection
  cleanup on the old instance may leak.

## How we'd find out

- Build a representative scene (1 deformer, 1 curve, small mesh).
  Save. Close the editor. Reopen. Verify everything looks identical.
- Modify the curve, save, close, reopen — same.
- Toggle `use_direct_delta_mush` on, save, close, reopen — DDM still
  active? Influences re-harvested or treated as cached?
- Open the same scene twice in different Godot instances; edit in
  one, observe whether the other detects the change.

## Mitigation if it breaks

- Explicit `_get_property_list` override that excludes mutable cache
  state from serialization (probably already implicit; verify).
- A custom resource for the rest cache (sidecar file) — see
  [known unknowns 08](../known_unknowns/08_bind_cache_persistence.md).
- Document any sharp edges (e.g. "do not share Curve3D resources
  across deformers") in the artist-facing docs.
