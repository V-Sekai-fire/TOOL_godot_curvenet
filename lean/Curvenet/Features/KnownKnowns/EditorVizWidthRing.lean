import Curvenet.IntersectionFrames

namespace Curvenet.Features.KnownKnowns

open Curvenet
open Curvenet.IntersectionFrames

/-- Feature 11.5 — Editor viz: width ring. Cyan octagon perpendicular
    to tangent at each knot, radius scaled by per-knot width `w`. The
    ring's radial scaling is consistent with `scaledFrame`'s width
    column (b · w): with t = +x, n = +z, l = 1, w = 1 the resulting
    frame is identity, so a width = 1 ring sits on a unit-radius
    octagon as drawn. -/
example :
    let M := scaledFrame ⟨1.0, 0.0, 0.0⟩ ⟨0.0, 0.0, 1.0⟩ 1.0 1.0
    -- B S column 0 = (1, 0, 0); column 1 (b · w) = (0, 1, 0); column 2 (n · h) = (0, 0, 1).
    ((M[0]! - 1.0).abs < 1e-12 ∧ M[1]!.abs < 1e-12 ∧ M[2]!.abs < 1e-12 ∧
     M[3]!.abs < 1e-12 ∧ (M[4]! - 1.0).abs < 1e-12 ∧ M[5]!.abs < 1e-12 ∧
     M[6]!.abs < 1e-12 ∧ M[7]!.abs < 1e-12 ∧ (M[8]! - 1.0).abs < 1e-12)
    := by native_decide

end Curvenet.Features.KnownKnowns
