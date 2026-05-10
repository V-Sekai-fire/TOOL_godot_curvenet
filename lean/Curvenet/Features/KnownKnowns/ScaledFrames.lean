import Curvenet.ScaledFrames

namespace Curvenet.Features.KnownKnowns

open Curvenet

/-- Feature 06 — Scaled frames + intersection frames. DeGoes22 §3
    deformation gradient is identity for rest = posed. -/
example :
    ScaledFrames.mat3WithinEps
      (ScaledFrames.deformationGradient ScaledFrames.mat3Identity ScaledFrames.mat3Identity)
      ScaledFrames.mat3Identity 1e-12 = true := by
  native_decide

end Curvenet.Features.KnownKnowns
