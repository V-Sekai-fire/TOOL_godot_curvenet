namespace Curvenet.Features.UnknownUnknowns

/-- Risk 06 — Numerical robustness on degenerate inputs. Sharp-Crane
    handles thin triangles; truly degenerate (zero edges, NaN coords,
    near-coincident knots) untested. -/
axiom degenerateInputs : True

end Curvenet.Features.UnknownUnknowns
