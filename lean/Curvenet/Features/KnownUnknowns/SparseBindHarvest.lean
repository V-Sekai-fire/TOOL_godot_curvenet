namespace Curvenet.Features.KnownUnknowns

/-- Gap 04 — Sparse bind harvest. DDM bind currently uses dense
    `harmonic_solve::solve_multi` (~30 s on 81 k); replace with
    sparse multi-RHS PCG over the cached `Lh_csr`. -/
axiom sparseBindHarvest : True

end Curvenet.Features.KnownUnknowns
