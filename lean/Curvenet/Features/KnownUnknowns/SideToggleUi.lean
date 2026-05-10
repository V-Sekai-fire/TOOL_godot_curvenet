namespace Curvenet.Features.KnownUnknowns

/-- Gap 05 — Side toggle UI. `IntersectionFrames::per_side_scaled_frames`
    already produces plus / minus sides but the deformer always pulls
    from one; expose a per-knot side flag + click-to-flip marker at
    intersection knots. -/
axiom sideToggleUi : True

end Curvenet.Features.KnownUnknowns
