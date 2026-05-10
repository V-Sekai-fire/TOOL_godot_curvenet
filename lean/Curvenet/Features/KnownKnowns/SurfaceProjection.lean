import Curvenet.SurfaceProjection

namespace Curvenet.Features.KnownKnowns

open Curvenet

/-- Feature 04 — Surface projection (vertex-only). Knot at the
    origin projects to vertex 0 of the triangle mesh. -/
example :
    let origin : Vec3 := ⟨0.0, 0.0, 0.0⟩
    let triMesh : Array Vec3 :=
      #[origin, ⟨1.0, 0.0, 0.0⟩, ⟨0.0, 1.0, 0.0⟩]
    let r := SurfaceProjection.projectToVertices #[origin] triMesh
    r.size = 1 ∧ r[0]!.meshIndex = 0 := by native_decide

end Curvenet.Features.KnownKnowns
