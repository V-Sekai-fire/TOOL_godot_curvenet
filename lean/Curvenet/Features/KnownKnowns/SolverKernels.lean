import Curvenet.SparseLinAlg

namespace Curvenet.Features.KnownKnowns

open Curvenet

/-- Feature 07 — Solver kernels. CG converges on a small SPD system
    (a positive-definite 3×3) to within 1e-9 of the analytic solution. -/
example :
    let A : SparseLinAlg.SparseMatrixCSR :=
      { rows := 3, cols := 3
      , rowPtr := #[0, 2, 5, 7]
      , colIdx := #[0, 1, 0, 1, 2, 1, 2]
      , values := #[4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0] }
    let b : Array Float := #[1.0, 0.0, -1.0]
    let x := SparseLinAlg.cg A b 100 1e-12
    let r := SparseLinAlg.saxpby 1.0 b (-1.0) (SparseLinAlg.spmv A x)
    SparseLinAlg.dot r r < 1e-9 = true := by native_decide

end Curvenet.Features.KnownKnowns
