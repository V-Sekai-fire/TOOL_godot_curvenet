namespace Curvenet.Features.KnownUnknowns

/-- Gap 03 — Quest 3 GPU dispatch. DDM runtime kernel is CPU; port
    `direct_delta_mush::lbs_matvec` to a Vulkan compute shader for
    the 0.8 ms Quest 3 budget. -/
axiom quest3GpuDispatch : True

end Curvenet.Features.KnownUnknowns
