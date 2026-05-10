import Curvenet.CutMeshLaplacian
import Curvenet.PolygonLaplacian

namespace Curvenet.Features.KnownKnowns

open Curvenet

/-- Feature 05 — Cot-Laplacian. `VᵀLₕV` is symmetric on the cut-mesh
    `triangleWithSample`. Sharp-Crane mollified. -/
example :
    let triPositions : Array Vec3 :=
      #[⟨0.0, 0.0, 0.0⟩, ⟨1.0, 0.0, 0.0⟩, ⟨0.5, 0.0, 0.8660254037844386⟩]
    let M := CutMeshLaplacian.assembleVtLhV CutExamples.triangleWithSample triPositions
    PolygonLaplacian.isSymmetricWithin M 3 1e-10 = true := by native_decide

end Curvenet.Features.KnownKnowns
