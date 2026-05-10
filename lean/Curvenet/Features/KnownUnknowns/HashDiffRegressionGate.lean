namespace Curvenet.Features.KnownUnknowns

/-- Gap 14 — Hash-diff regression gate (ufbx-style). FNV-1a hashing
    of deformed positions + DDM influences against a baseline; tagged
    dump on mismatch with parent-stack breadcrumbs. -/
axiom hashDiffRegressionGate : True

end Curvenet.Features.KnownUnknowns
