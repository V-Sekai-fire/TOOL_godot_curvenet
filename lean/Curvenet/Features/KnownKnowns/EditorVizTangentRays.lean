import Curvenet.IntersectionFrames

namespace Curvenet.Features.KnownKnowns

open Curvenet
open Curvenet.IntersectionFrames

/-- Feature 11.2 — Editor viz: tangent rays at intersection knots.
    Short yellow segments along each outgoing tangent. The renderer
    consults `cornerNormals`; at a perpendicular 2-tangent
    intersection in the xy-plane the corner normals point along ±z
    with unit length, so the rays are well-defined and length-preserving. -/
example :
    let segs : Array OutgoingSegment :=
      #[ ⟨⟨1.0, 0.0, 0.0⟩, 1.0⟩
       , ⟨⟨0.0, 1.0, 0.0⟩, 1.0⟩ ]
    let ms := cornerNormals segs
    let m0 := ms[0]!
    ((m0.x).abs < 1e-12 ∧ (m0.y).abs < 1e-12 ∧ (m0.z - 1.0).abs < 1e-12)
    := by native_decide

end Curvenet.Features.KnownKnowns
