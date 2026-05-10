import Curvenet.CurvenetBuilder

namespace Curvenet.Features.KnownKnowns

open Curvenet

/-- Feature 11.1 — Editor viz: knot-kind markers. Coloured "+"-cross
    per merged knot, bucketed by `KnotKind` (anchor red, regular
    white, intersection green). The bucket choice is driven by
    `CurvenetBuilder.classify`; on a two-knot open curve both
    endpoints classify as anchors. -/
example :
    let A : Vec3 := ⟨0.0, 0.0, 0.0⟩
    let B : Vec3 := ⟨1.0, 0.0, 0.0⟩
    let g := CurvenetBuilder.build #[#[A, B]] 1e-6
    CurvenetBuilder.classify g
      = #[CurvenetBuilder.KnotKind.anchor, CurvenetBuilder.KnotKind.anchor]
    := by native_decide

end Curvenet.Features.KnownKnowns
