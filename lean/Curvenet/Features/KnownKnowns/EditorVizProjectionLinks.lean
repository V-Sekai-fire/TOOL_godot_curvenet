import Curvenet.SurfaceProjection

namespace Curvenet.Features.KnownKnowns

open Curvenet

/-- Feature 11.6 — Editor viz: projection links. Cyan lines from each
    merged knot to its closest mesh vertex. The renderer calls
    `SurfaceProjection.projectToVertices` against the source mesh's
    vertex array; a knot at the origin of a triangle mesh whose first
    vertex is the origin projects to vertex 0. -/
example :
    let origin : Vec3 := ⟨0.0, 0.0, 0.0⟩
    let triMesh : Array Vec3 :=
      #[origin, ⟨1.0, 0.0, 0.0⟩, ⟨0.0, 1.0, 0.0⟩]
    let r := SurfaceProjection.projectToVertices #[origin] triMesh
    r[0]!.meshIndex = 0 := by native_decide

end Curvenet.Features.KnownKnowns
