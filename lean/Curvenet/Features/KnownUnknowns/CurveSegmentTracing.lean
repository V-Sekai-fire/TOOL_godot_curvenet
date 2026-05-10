namespace Curvenet.Features.KnownUnknowns

/-- Gap 02 — Curve-segment tracing through faces. Once edge/face
    projection lands (gap 01), curves still need a tracer that
    inserts cracks via `cut_algorithm::insert_crack_at_endpoint`
    along the geodesic between adjacent knots. Blocked by 01. -/
axiom curveSegmentTracing : True

end Curvenet.Features.KnownUnknowns
