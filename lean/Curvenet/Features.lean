import Curvenet.Features.KnownKnowns.AuthoringUx
import Curvenet.Features.KnownKnowns.CotLaplacian
import Curvenet.Features.KnownKnowns.CurvenetGraph
import Curvenet.Features.KnownKnowns.CutMesh
import Curvenet.Features.KnownKnowns.Degoes22Solve
import Curvenet.Features.KnownKnowns.DirectDeltaMush
import Curvenet.Features.KnownKnowns.EditorVizFrameAxes
import Curvenet.Features.KnownKnowns.EditorVizKnotKindMarkers
import Curvenet.Features.KnownKnowns.EditorVizProjectionLinks
import Curvenet.Features.KnownKnowns.EditorVizTangentControlLinks
import Curvenet.Features.KnownKnowns.EditorVizTangentRays
import Curvenet.Features.KnownKnowns.EditorVizWidthRing
import Curvenet.Features.KnownKnowns.HalfedgeMesh
import Curvenet.Features.KnownKnowns.ScaledFrames
import Curvenet.Features.KnownKnowns.SolverKernels
import Curvenet.Features.KnownKnowns.SurfaceProjection
import Curvenet.Features.KnownKnowns.TestInfrastructure
import Curvenet.Features.KnownUnknowns.BindCachePersistence
import Curvenet.Features.KnownUnknowns.CurveSegmentTracing
import Curvenet.Features.KnownUnknowns.EdgeFaceProjection
import Curvenet.Features.KnownUnknowns.GodotCsgIntegration
import Curvenet.Features.KnownUnknowns.HashDiffRegressionGate
import Curvenet.Features.KnownUnknowns.PerKnotLDrag
import Curvenet.Features.KnownUnknowns.Quest3GpuDispatch
import Curvenet.Features.KnownUnknowns.SideToggleUi
import Curvenet.Features.KnownUnknowns.SparseBindHarvest
import Curvenet.Features.KnownUnknowns.SurfaceNormalReferenceFrame

/-!
# Features manifest

Umbrella import. Each feature and gap lives in its own file under
`Features/{KnownKnowns, KnownUnknowns}/`. Known knowns are
`native_decide` proofs of canonical claims; known unknowns are
`axiom name : True` placeholders with docstring-encoded semantics.
Replace an axiom with `def name : True := trivial` when its gap
closes.

Project policy: zero `sorry` / `admit`.
-/
