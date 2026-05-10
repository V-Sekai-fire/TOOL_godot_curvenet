import Curvenet.Features.KnownKnowns.AuthoringUx
import Curvenet.Features.KnownKnowns.CotLaplacian
import Curvenet.Features.KnownKnowns.CurvenetGraph
import Curvenet.Features.KnownKnowns.CutMesh
import Curvenet.Features.KnownKnowns.Degoes22Solve
import Curvenet.Features.KnownKnowns.DirectDeltaMush
import Curvenet.Features.KnownKnowns.EditorVisualization
import Curvenet.Features.KnownKnowns.HalfedgeMesh
import Curvenet.Features.KnownKnowns.ScaledFrames
import Curvenet.Features.KnownKnowns.SolverKernels
import Curvenet.Features.KnownKnowns.SurfaceProjection
import Curvenet.Features.KnownKnowns.TestInfrastructure
import Curvenet.Features.KnownUnknowns.AnimatedDrag
import Curvenet.Features.KnownUnknowns.AutoCurveExtraction
import Curvenet.Features.KnownUnknowns.BindCachePersistence
import Curvenet.Features.KnownUnknowns.CurveSegmentTracing
import Curvenet.Features.KnownUnknowns.EdgeFaceProjection
import Curvenet.Features.KnownUnknowns.GodotCsgIntegration
import Curvenet.Features.KnownUnknowns.HashDiffRegressionGate
import Curvenet.Features.KnownUnknowns.HeatMethodPolygonSoups
import Curvenet.Features.KnownUnknowns.MultiCharacterBatching
import Curvenet.Features.KnownUnknowns.PerKnotLDrag
import Curvenet.Features.KnownUnknowns.Quest3GpuDispatch
import Curvenet.Features.KnownUnknowns.SideToggleUi
import Curvenet.Features.KnownUnknowns.SparseBindHarvest
import Curvenet.Features.KnownUnknowns.SurfaceNormalReferenceFrame
import Curvenet.Features.UnknownUnknowns.AnimationPoseSpace
import Curvenet.Features.UnknownUnknowns.DegenerateInputs
import Curvenet.Features.UnknownUnknowns.GodotVersionDrift
import Curvenet.Features.UnknownUnknowns.MultiDeformerScene
import Curvenet.Features.UnknownUnknowns.MultiPlatformBuilds
import Curvenet.Features.UnknownUnknowns.RealCharacterRigBeyondMire
import Curvenet.Features.UnknownUnknowns.ScalingBeyondBenchmark
import Curvenet.Features.UnknownUnknowns.SceneSaveLoadLifecycle

/-!
# Features manifest

Umbrella import. Each feature, gap, and risk lives in its own file
under `Features/{KnownKnowns, KnownUnknowns, UnknownUnknowns}/`.
Known knowns are `native_decide` proofs of canonical claims; known
unknowns and unknown unknowns are `axiom name : True` placeholders
with docstring-encoded semantics. Replace an axiom with a
`def name : True := trivial` when its gap closes.

Project policy: zero `sorry` / `admit`.
-/
