/-
Lean4 formalization of the CurveNet math layer.

Mirrors the rapidcheck property tests in tests/. Proofs here are over real-valued
3-vectors and the cubic Bezier / Coons patch constructions in src/curvenet/.

Status:
- bezierBasis sums to 1 ........................... ✓ proven
- bezier 0 = p0, bezier 1 = p3 .................... ✓ proven
- coincident control points → constant ............ ✓ proven
- coons evaluates corners exactly ................. ✓ proven
- coons recovers u0/u1/v0/v1 boundaries ........... ✓ proven (translation-invariant by partition of unity)
- translation invariance of bezier ................ ✓ proven
- N=4 NgonPatch ↔ CoonsPatch equivalence .......... TODO (definitional once NgonPatch is encoded)
-/

import Curvenet.Vec3
import Curvenet.Bezier
import Curvenet.CoonsPatch
import Curvenet.NgonPatch
import Curvenet.MeanValue
import Curvenet.Halfedge
import Curvenet.PolygonLaplacian
import Curvenet.CutMesh
