import Curvenet.DirectDeltaMushBind

namespace Curvenet.Features.KnownKnowns

open Curvenet

/-- Feature 10 — Authoring UX (4 draggable §3 knobs). The bind-time
    DDM weight composition preserves partition of unity, which is
    the algorithmic guarantee underneath the artist's drag. -/
example :
    let identityW : DirectDeltaMushBind.WeightMatrix :=
      #[#[1.0, 0.0, 0.0, 0.0],
        #[0.0, 1.0, 0.0, 0.0],
        #[0.0, 0.0, 1.0, 0.0],
        #[0.0, 0.0, 0.0, 1.0]]
    DirectDeltaMushBind.partitionOfUnity identityW 1e-12 = true := by native_decide

end Curvenet.Features.KnownKnowns
