import Curvenet.SlangCodegen.Common
import Curvenet.SlangCodegen.DirectDeltaMush
import Curvenet.SlangCodegen.Axpy
import Curvenet.SlangCodegen.AxpyMulti
import Curvenet.SlangCodegen.Saxpby
import Curvenet.SlangCodegen.SaxpbyMulti
import Curvenet.SlangCodegen.Jacobi
import Curvenet.SlangCodegen.JacobiMulti
import Curvenet.SlangCodegen.Spmv
import Curvenet.SlangCodegen.SpmvMulti
import Curvenet.SlangCodegen.DotReduce
import Curvenet.SlangCodegen.DotReduceMulti
import Curvenet.SlangCodegen.SgsColor
import Curvenet.SlangCodegen.PolygonLaplacian
import Curvenet.SlangCodegen.RobustLaplacian
import Curvenet.SlangCodegen.ScaledFrames
import Curvenet.SlangCodegen.SegmentGradient
import Curvenet.SlangCodegen.IntersectionFrames
import Curvenet.SlangCodegen.CurveInterp
import Curvenet.SlangCodegen.Vec3
import Curvenet.SlangCodegen.Halfedge
import Curvenet.SlangCodegen.HalfedgeBuilder
import Curvenet.SlangCodegen.CutMesh
import Curvenet.SlangCodegen.CutAlgorithm
import Curvenet.SlangCodegen.SurfaceProjection
import Curvenet.SlangCodegen.CurvenetBuilder
import Curvenet.SlangCodegen.IncompleteCholesky
import Curvenet.SlangCodegen.HierarchicalSparsify
import Curvenet.SlangCodegen.CutMeshLaplacian
import Curvenet.SlangCodegen.DenseLinAlg
import Curvenet.SlangCodegen.SparseLinAlg
import Curvenet.SlangCodegen.HarmonicSolve
import Curvenet.SlangCodegen.DeformSolve

/-!
# `Curvenet.SlangCodegen` — Slang shader codegen umbrella

Each submodule produces a `LeanSlang.SlangShaderModule` for one of
the GPU kernels Curvenet's deformer needs. Pinned `native_decide`
fixtures assert the emission text against a hand-checked reference
per kernel.

Layout convention: every kernel module exports

- `shader   : SlangShaderModule`
- `expected : String`
- two `example` lemmas: `emit shader = expected` and
  `shader.entryPointName = "main"`.
-/
