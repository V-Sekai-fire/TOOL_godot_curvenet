# 12 — Animated knot drag

## Why (current pain)

Authoring is real-time-only — once an artist drags knots, the result
is the new pose, not a recording. There's no way to keyframe knot
positions / tilts / widths over time and replay them as an animation
track. Low/research priority gap.

## Gall-minimum slice

- **In scope**: hook `EditorUndoRedoManager` actions into a
  `AnimationPlayer` track when the user is in "record" mode. Each
  knot drag becomes a keyframe of the affected property at the
  current time cursor.
- **Deferred**: cross-curve interpolation (Godot AnimationPlayer
  handles per-property interpolation already); programmatic
  authoring of curves via tweens.
- **Why this slice**: piggybacks on Godot's native AnimationPlayer
  rather than inventing a parallel system. Tiny conceptual addition,
  big artistic value.

## Files to touch

- `src/curvenet_animation.h` (new) — small recorder helper that
  exposes a "record / stop" toggle on the deformer.
- `src/curvenet_animation.cpp` (new) — when recording, every
  `_commit_handle` call inserts a track keyframe via
  `AnimationPlayer::insert_track_key`.
- `src/curvenet_gizmo_plugin.cpp` — when recording, the existing
  `_commit_handle` callback calls into the recorder before the normal
  undo/redo registration.
- `src/curvenet_editor_plugin.cpp` — toolbar button to toggle
  recording mode.

## Approach

- Use Godot's native `AnimationPlayer` and `Animation` resources —
  no custom format. Track names match property paths
  (`profile_curves/0/point_position/3`,
  `knot_widths/0/3`, `profile_curves/0/point_tilt/3`).
- "Record" toggle adds a child `AnimationPlayer` to the deformer
  if one isn't present, picks the current animation in the
  AnimationPlayer (default: a "rest" one), then inserts keyframes on
  drag commits.
- Existing undo/redo behavior is unaffected; the keyframe insertion
  is an additional `add_do_method` call when recording is active.
- Playback uses Godot's standard AnimationPlayer scrubber — no new
  UI.

## Verification

- **Lean**: no change.
- **C++**: ~3 RC properties on the recorder's track-insertion
  bookkeeping.
- **GDExtension**: `scons -j8` clean.
- **Manual**: enable recording, drag a knot, advance time, drag
  again. Hit play — the deformer should animate between the two
  poses.

## Blocks / blocked-by

- **Blocks**: nothing.
- **Blocked-by**: nothing.

## Estimated cost

- LOC: ~400 across animation header (~150) + impl (~150) + gizmo
  hook (~50) + plugin button (~50).
- Effort: large (mostly Godot AnimationPlayer API surface, not new
  algorithm).
- Risk: low. Godot's AnimationPlayer is well-trodden; the only
  novelty is wiring the recorder into the existing drag commit path.
