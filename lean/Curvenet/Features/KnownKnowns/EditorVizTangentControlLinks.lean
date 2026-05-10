import Curvenet.CurvenetBuilder

namespace Curvenet.Features.KnownKnowns

open Curvenet

/-- Feature 11.3 — Editor viz: tangent control links. Lines from
    each Curve3D anchor to its `point_in` / `point_out`. The
    renderer enumerates `(curve_id, knot_idx)` references through
    `CurvenetGraph.incidence`; on a single open 2-knot curve the
    incidence list at each knot has exactly one entry, giving exactly
    two `point_in/point_out` link pairs in the gizmo. -/
example :
    let A : Vec3 := ⟨0.0, 0.0, 0.0⟩
    let B : Vec3 := ⟨1.0, 0.0, 0.0⟩
    let g := CurvenetBuilder.build #[#[A, B]] 1e-6
    g.incidence.size = 2 ∧
    g.incidence[0]!.size = 1 ∧
    g.incidence[1]!.size = 1
    := by native_decide

end Curvenet.Features.KnownKnowns
