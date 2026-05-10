# 03 — Quest 3 GPU dispatch for DDM matvec

## Why (current pain)

DDM runtime kernel is CPU-only. At 50 k vertices × 8 influences per
vertex × 90 Hz, the matvec doesn't fit in Quest 3's CPU budget — but
its memory-bandwidth footprint (~1.6 MB read traffic per frame) is
well inside the GPU's envelope. Without GPU dispatch the project
can't ship to Quest 3, which is the entire reason DDM was chosen over
the §6 path. #3 high-priority gap in
[`docs/FEATURES.md`](../docs/FEATURES.md#high-priority).

## Gall-minimum slice

- **In scope**: Vulkan compute shader port of
  `direct_delta_mush::lbs_matvec`. One workgroup per vertex (or one
  thread per vertex with local_size_x = 64), reads sparse influence
  list + per-handle 4×4 from SSBOs, writes deformed position to an
  output buffer.
- **Deferred**: GPU bind harvest (still CPU; bake-once, dispatch-many);
  multi-character batched dispatch (todo 11); GPU-side Curve3D state
  capture (artist drags happen on CPU, transforms are uploaded each
  frame).
- **Why this slice**: ports the *runtime* matvec only — the bind
  pipeline can stay CPU and just upload the resulting weight rows
  once per topology change.

## Files to touch

- `src/curvenet/shaders/ddm_matvec.comp` (new) — the compute shader.
  Bindings: `params` (UBO), `influence_row_ptr`, `influence_col_idx`,
  `influence_weights`, `transforms`, `rest_positions`,
  `out_positions`.
- `src/curvenet_deformer_3d.cpp` — when DDM is active and a GPU
  device is available, dispatch the shader instead of running the CPU
  matvec. Keep CPU as fallback.
- `src/gpu_compute_helpers.h` — reuse buffer creation / SPIR-V
  dispatch helpers (already used by `sgs_color.comp`).
- `tests/bench_ddm_70k_gpu.cpp` (new) — benchmark CPU vs GPU on the
  Mire 70 k mesh.

## Approach

- Compile shader to SPIR-V via `glslangValidator` at build time
  (precedent: `shaders/sgs_color.comp` in the build pipeline).
- Pack the per-vertex sparse influence list as CSR (row_ptr, col_idx,
  weights). Already the natural format from
  `direct_delta_mush::sparsify_top_k` output — write a small bridge
  to CSR.
- Per dispatch, upload `transforms` SSBO (16 floats per handle) and
  reuse cached buffers for the rest.
- Output buffer is read back via `vkMapMemory` or kept on-GPU and
  fed directly to a `MeshInstance3D` vertex buffer (the latter is the
  Quest 3 win — zero CPU readback).
- Validate parity with CPU on a fixture mesh: same input → same output
  to 1e-5 (float vs double precision).

## Verification

- **Lean**: no change (shader code is GPU side).
- **C++**: `tests/bench_ddm_70k_gpu.cpp` measures GPU dispatch time;
  parity test asserts CPU/GPU outputs agree to eps.
- **GDExtension**: `scons -j8` builds cleanly with the new shader
  compilation step.
- **Manual**: on Quest 3 build, profile the runtime path with
  RenderDoc or Quest's GPU profiler. Target: < 0.8 ms / frame at
  50 k.

## Blocks / blocked-by

- **Blocks**: Quest 3 ship.
- **Blocked-by**: none (the CPU DDM path is fully wired and ready
  to mirror).

## Estimated cost

- LOC: ~500 across shader (~80) + dispatch glue (~250) + buffer
  setup (~120) + bench (~50).
- Effort: medium-large.
- Risk: medium. GPU debugging is harder than CPU; SPIR-V toolchain
  on macOS dev / Linux CI / Quest 3 target needs to be sanity-checked.
