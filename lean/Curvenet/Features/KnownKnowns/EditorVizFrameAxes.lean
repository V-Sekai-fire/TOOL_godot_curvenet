import Curvenet.ScaledFrames

namespace Curvenet.Features.KnownKnowns

open Curvenet

/-- Feature 11.4 — Editor viz: per-knot frame axes
    (red tangent / green normal / blue binormal). The runtime frame
    derivation that the gizmo mirrors is the same
    `isolatedSegmentGradient` proven for rest = posed → identity. -/
example :
    ScaledFrames.mat3WithinEps
      (ScaledFrames.isolatedSegmentGradient ⟨0,0,0⟩ ⟨1,0,0⟩ ⟨0,0,0⟩ ⟨1,0,0⟩)
      ScaledFrames.mat3Identity 1e-12 = true := by
  native_decide

end Curvenet.Features.KnownKnowns
