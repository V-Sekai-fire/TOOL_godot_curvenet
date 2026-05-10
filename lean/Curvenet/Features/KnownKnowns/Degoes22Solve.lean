import Curvenet.DeformSolve

namespace Curvenet.Features.KnownKnowns

open Curvenet

/-- Feature 08 — DeGoes22 §6 two-stage solve. Pure translation:
    `f_c = identity`, `x_c = (5, 0, 0)` on `triangleWithSample` →
    vertex 1 ends at (6, 0, 0). -/
example :
    let positions : Array Vec3 :=
      #[ ⟨0.0, 0.0, 0.0⟩, ⟨1.0, 0.0, 0.0⟩, ⟨0.5, 0.0, 0.8660254037844386⟩ ]
    let oneSample : Nat → Nat → Bool → Nat := fun _ _ _ => 0
    let Fc : DenseLinAlg.Mat :=
      #[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    let Xc : DenseLinAlg.Mat := #[5.0, 0.0, 0.0]
    let Xv := DeformSolve.solveDeformation CutExamples.triangleWithSample positions oneSample Fc Xc
    let v1 : Array Float := #[DenseLinAlg.get Xv 3 1 0,
                               DenseLinAlg.get Xv 3 1 1,
                               DenseLinAlg.get Xv 3 1 2]
    DenseLinAlg.vecWithinEps v1 #[6.0, 0.0, 0.0] 1e-9 = true := by
  native_decide

end Curvenet.Features.KnownKnowns
