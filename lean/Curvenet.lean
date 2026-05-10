/-
Lean4 formalization of the CurveNet math layer (DeGoes22 + slicing).

Mirrors the per-slice C++ runtime under `src/curvenet/`. Each module
contains its own `native_decide` instance proofs of the algorithm's
load-bearing invariants on small concrete examples.

Modules organised by §:
  Vec3, ScaledFrames, IntersectionFrames, CurveInterp, SegmentGradient
                                  — DeGoes22 §3 curvenet representation
  Halfedge, PolygonLaplacian, CutMesh, CutMeshLaplacian, CutAlgorithm
                                  — §4.1 cut-mesh + §4.2 discretization
  DenseLinAlg, SparseLinAlg       — solver kernels (LU + sparse CG)
  HarmonicSolve, DeformSolve      — §4.3 two-stage solve
-/

import Curvenet.Common
import Curvenet.Vec3
import Curvenet.Halfedge
import Curvenet.HalfedgeBuilder
import Curvenet.CurvenetBuilder
import Curvenet.SurfaceProjection
import Curvenet.PolygonLaplacian
import Curvenet.RobustLaplacian
import Curvenet.CutMesh
import Curvenet.ScaledFrames
import Curvenet.DenseLinAlg
import Curvenet.SparseLinAlg
import Curvenet.CutMeshLaplacian
import Curvenet.HarmonicSolve
import Curvenet.DeformSolve
import Curvenet.CutAlgorithm
import Curvenet.IntersectionFrames
import Curvenet.CurveInterp
import Curvenet.SegmentGradient
import Curvenet.IncompleteCholesky
import Curvenet.HierarchicalSparsifyCompensate
import Curvenet.GraphColoring
import Curvenet.DirectDeltaMush
import Curvenet.DirectDeltaMushBind
import Curvenet.SlangCodegen
import Curvenet.Features
