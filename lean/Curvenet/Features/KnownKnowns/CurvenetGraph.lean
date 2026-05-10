import Curvenet.CurvenetBuilder

namespace Curvenet.Features.KnownKnowns

/-- Feature 02 — Curvenet graph. ε-merge + classify. Two open
    curves sharing an endpoint within ε produce 3 distinct merged
    knots. -/
example :
    let A : Curvenet.Vec3 := ⟨0.0, 0.0, 0.0⟩
    let B : Curvenet.Vec3 := ⟨1.0, 0.0, 0.0⟩
    let Bp : Curvenet.Vec3 := ⟨1.0 + 1e-10, 0.0, 0.0⟩
    let C : Curvenet.Vec3 := ⟨2.0, 0.0, 0.0⟩
    let g := Curvenet.CurvenetBuilder.build #[#[A, B], #[Bp, C]] 1e-6
    g.knotPositions.size = 3 := by native_decide

end Curvenet.Features.KnownKnowns
