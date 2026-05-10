# 01 — Real character rig validation

## What we don't know

The deformer has been benchmarked against a single mesh family (Mire
body at 5k / 81k vertices) with synthetic profile curves authored to
exercise the algorithm. It has **never been run on an artist-authored
production character rig** with the deformer driving the actual
character pose. We don't know what breaks when the input is not
synthesized.

## What might break

- Curve placement patterns artists actually use (long thin profiles
  along limbs, dense intersections at joints, very short segments
  near eyelids) may surface numerical / topology cases that our
  fixtures don't reach.
- Real character meshes have UV seams, multiple submeshes, blend
  shapes, vertex groups — none of which the deformer currently
  consults. Blend shapes may interact badly with the bind cache.
- Production rigs typically include skinning weights from a
  separate rig (LBS bones). Layering the curvenet deformer on top is
  untested — the order of operations matters.
- The §6 solve assumes a manifold input. fTetWild integration (per
  the manifold-prepass memory) is planned but not yet wired; in the
  meantime, real character meshes with non-manifold edges may fail
  CG silently.

## How we'd find out

- Pick one open-source character rig (e.g. V-Sekai's avatar set, or a
  Mixamo character). Wire it into a test scene with hand-authored
  profile curves at typical joints (shoulder, hip, knee, elbow).
- Drag a curve. Observe deformation quality. Compare side-by-side
  against the same character with traditional LBS skinning.
- Capture failure modes: NaN positions, exploding deformations,
  silent divergence, memory blowup, FPS drop.
- Repeat for at least 3 different characters. Patterns that recur
  across rigs are real.

## Mitigation if it breaks

- Add the failing character to `tests/` as a fixture (commit the mesh
  + curves into the repo).
- File a [`docs/features/known_unknowns/`](../known_unknowns/) entry
  if the failure mode generalizes.
- Each surprise turns into a known unknown — that's the point of
  writing this file: we track that we *will* find some, even if we
  don't know which yet.
