# 01 — Real character rig validation

## What we don't know

The deformer has been benchmarked against the **Mire body** rig
([V-Sekai-fire/mesh-mille-mire-feuille](https://github.com/V-Sekai-fire/mesh-mille-mire-feuille))
at 5 k and 81 k vertices — a real production rig, not a synthetic
fixture. What's still untested:

- Real *artist-authored* profile curves (we've only used hand-crafted
  curves to exercise the algorithm; no artist has driven Mire
  end-to-end through the deformer yet).
- More than one rig — Mire is one production rig family; we don't
  know how the assumptions baked into its topology generalise.
- Integration with Mire's other deformation layers (LBS bones,
  blend shapes, etc.) when the curvenet deformer is layered on top.

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

- Hand Mire to an artist with no algorithmic context. Have them
  author profile curves at typical joints (shoulder, hip, knee,
  elbow). Watch what they expect vs what the deformer delivers.
- Add a second rig (V-Sekai's avatar set, a Mixamo character, etc.)
  to compare what's Mire-specific vs what generalises.
- Drag a curve. Observe deformation quality. Compare side-by-side
  against the same character with traditional LBS skinning.
- Capture failure modes: NaN positions, exploding deformations,
  silent divergence, memory blowup, FPS drop.
- Repeat for at least 3 different rigs. Patterns that recur across
  rigs are real production failure modes.

## Mitigation if it breaks

- Add the failing character to `tests/` as a fixture (commit the mesh
  + curves into the repo).
- File a [`docs/features/known_unknowns/`](../known_unknowns/) entry
  if the failure mode generalizes.
- Each surprise turns into a known unknown — that's the point of
  writing this file: we track that we *will* find some, even if we
  don't know which yet.
