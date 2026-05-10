import Curvenet.HalfedgeBuilder

namespace Curvenet.Features.KnownKnowns

open Curvenet.HalfedgeBuilderExamples

/-- Feature 01 — Halfedge mesh. Triangle-soup → manifold halfedge
    structure. Re-runs the canonical claim that
    `HalfedgeBuilder.fromTriangles` produces a manifold mesh on a
    single triangle. -/
example : singleTri.manifold? = true := by native_decide

end Curvenet.Features.KnownKnowns
