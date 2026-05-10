# 02 — Animation playback / pose-space behavior

## What we don't know

Today, `apply_deformation` runs on each artist drag in the editor.
We've never run the deformer with an `AnimationPlayer` driving the
profile curves' positions / tilts / widths over time. Pose-space
behavior — what the deformation looks like across an animated curve
sequence — is untested.

## What might break

- The runtime path's warm-start (PCG seeded with previous frame's
  iterate) assumes small frame-to-frame deltas. A keyframed
  animation that jumps 30° between frames may pessimize CG to
  cold-start cost or worse, oscillate across frames.
- DDM bind cache is invalidated only on topology change. Animated
  knot positions don't trigger rebind, but the harvested weights
  assumed a *specific* rest pose. If the artist animates the curve
  far from rest, the LBS-style matvec extrapolates linearly — wrong
  for large deviations.
- Curve3D's per-point properties are individually animatable. There's
  no "curvenet-frame" abstraction in Godot's animation system — every
  knot is a separate track.
- Frame-pacing cost: at 60 Hz, even the 5k §6 solve at 3-7 ms is
  half the frame budget on a single thread. Untested at AnimationPlayer
  speeds.

## How we'd find out

- Build a test scene: one character mesh, one Curve3D with 5 knots,
  one AnimationPlayer track keyframing each knot's position to make
  the curve oscillate.
- Play the animation, profile per-frame cost.
- Inspect the output: smooth deformation? Jumps at keyframes? CG
  divergence on big deltas?
- Toggle DDM on, repeat. Does the matvec pose-extrapolate gracefully
  or break down?

## Mitigation if it breaks

- Cache invalidation on knot-position-delta exceeding a threshold
  (rebuild DDM weights mid-animation if the artist animates far from
  rest).
- A "rest pose" override property so artists can decouple animated
  state from the bind reference.
- Documentation — animation pose-space is best-effort until proven.
