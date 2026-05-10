import Curvenet.CutMesh

namespace Curvenet.Features.KnownKnowns

open Curvenet

/-- Feature 03 — Cut-mesh + cut-algorithm. Sample-promoted vertices
    + surgery primitives. Re-runs the canonical partition-of-unity
    invariant on `triangleWithSample`. -/
example :
    let oneSample : Nat → Nat → Bool → Nat := fun _ _ _ => 0
    CutExamples.triangleWithSample.partitionOfUnity oneSample = true := by
  native_decide

end Curvenet.Features.KnownKnowns
