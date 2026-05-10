import Curvenet.DirectDeltaMush

namespace Curvenet.Features.KnownKnowns

open Curvenet

/-- Feature 09 — Direct Delta Mush. Identity transform with full
    weight = 1 preserves the rest position. -/
example :
    let infl : DirectDeltaMush.Influences := #[(0, 1.0)]
    let transforms : Array DirectDeltaMush.Mat4 := #[DirectDeltaMush.mat4Identity]
    DirectDeltaMush.vclose
      (DirectDeltaMush.lbsMatvec transforms infl 1.5 (-2.0) 0.75)
      (1.5, -2.0, 0.75) 1e-12 = true := by
  native_decide

end Curvenet.Features.KnownKnowns
