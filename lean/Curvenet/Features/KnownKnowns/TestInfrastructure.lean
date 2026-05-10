import Curvenet.Halfedge

namespace Curvenet.Features.KnownKnowns

open Curvenet

/-- Feature 12 — Test + spec infrastructure. The smallest manifold
    fixture in this library passes its manifold check, demonstrating
    the test/spec layer is functional. -/
example : Curvenet.Examples.triangle.manifold? = true := by native_decide

end Curvenet.Features.KnownKnowns
