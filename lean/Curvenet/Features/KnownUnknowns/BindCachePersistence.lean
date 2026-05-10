namespace Curvenet.Features.KnownUnknowns

/-- Gap 08 — Bind cache persistence. Every Godot session re-binds;
    serialize `RestCache` to a sidecar `.curvenet_cache.bin`. -/
axiom bindCachePersistence : True

end Curvenet.Features.KnownUnknowns
