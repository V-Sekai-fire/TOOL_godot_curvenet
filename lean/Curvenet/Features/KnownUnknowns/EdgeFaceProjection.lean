namespace Curvenet.Features.KnownUnknowns

/-- Gap 01 — Edge+face surface projection. Today vertex-only;
    `SurfaceProjection.ProjectionKind` reserves `edgeIntersection`
    and `faceInterior` but the producer never emits them. Closing
    this requires a closest-point-on-triangle helper +
    `promote_samples` dispatcher. -/
axiom edgeFaceProjection : True

end Curvenet.Features.KnownUnknowns
