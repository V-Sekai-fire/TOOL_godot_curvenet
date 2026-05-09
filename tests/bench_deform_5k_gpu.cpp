// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
//
// Bench the existing tests/gpu_cg_solver.h against the deformer's
// actual 81k LhsM matrix — not a synthetic 2D Laplacian. Per Gall's
// law: validate the GPU path on the real input before adding any
// meshlet / multi-RHS layering. If GPU CG produces sane convergence
// here at all, the architecture is sound on this matrix; speed is
// then a function of the underlying Vulkan transport (MoltenVK on
// the dev box, native Vulkan on Steam Deck / Quest 3).

#include "curvenet/cut_mesh.h"
#include "curvenet/cut_mesh_laplacian.h"
#include "curvenet/halfedge_builder.h"
#include "curvenet/sparse_linalg.h"
#include "curvenet/vec3.h"
#include "gpu_cg_solver.h"
#include "mire_body_data.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

using curvenet::Vec3;
namespace cm  = curvenet::cut_mesh;
namespace cml = curvenet::cut_mesh_laplacian;
namespace sp  = curvenet::sparse;
using namespace curvenet_gpu_test;

namespace {

double now_ms() {
    using namespace std::chrono;
    return duration_cast<duration<double, std::milli>>(
        steady_clock::now().time_since_epoch()).count();
}

int closest_vertex(const std::vector<Vec3> &positions, const Vec3 &target) {
    int best = 0;
    double best_d2 = 1e300;
    for (std::size_t i = 0; i < positions.size(); ++i) {
        const double dx = positions[i].x - target.x;
        const double dy = positions[i].y - target.y;
        const double dz = positions[i].z - target.z;
        const double d2 = dx * dx + dy * dy + dz * dz;
        if (d2 < best_d2) { best_d2 = d2; best = static_cast<int>(i); }
    }
    return best;
}

} // namespace

int main(int argc, char **argv) {
    (void)argc; (void)argv;

    std::printf("Mire body 5k: %d verts, %d tris\n",
                  mire_body::n_verts, mire_body::n_tris);

    std::vector<Vec3> positions;
    positions.reserve(mire_body::n_verts);
    for (int i = 0; i < mire_body::n_verts; ++i) {
        positions.push_back({
            static_cast<double>(mire_body::positions[i * 3 + 0]),
            static_cast<double>(mire_body::positions[i * 3 + 1]),
            static_cast<double>(mire_body::positions[i * 3 + 2]),
        });
    }
    std::vector<int> tris(mire_body::tris,
                            mire_body::tris + mire_body::n_tris * 3);
    const std::size_t nv = positions.size();

    const curvenet::HalfedgeMesh hm =
        curvenet::halfedge_builder::from_triangles(nv, tris);
    cm::CutMesh c;
    c.base = hm;
    c.vertex_kind.assign(nv, cm::CutVertexKind::mesh_vertex_kind());
    c.segment_of_halfedge.assign(hm.he_count(), -1);
    const std::vector<Vec3> sample_targets = {
        {  0.4,  0.0, 1.0 }, { -0.4,  0.0, 1.0 },
        {  0.0,  0.0, 1.6 }, {  0.0,  0.0, 0.6 },
    };
    for (std::size_t s = 0; s < sample_targets.size(); ++s) {
        const int vi = closest_vertex(positions, sample_targets[s]);
        c.vertex_kind[static_cast<std::size_t>(vi)] =
            cm::CutVertexKind::sample_kind(static_cast<int>(s), 0, false);
    }

    const double mol_delta = cml::default_mollify_delta(positions, tris);
    const double t_bind = now_ms();
    sp::SparseMatrixCSR LhsM =
        cml::assemble_vt_lh_v_csr_robust(c, positions, mol_delta);
    std::printf("LhsM bind: %.0f ms  (rows=%zu nnz=%zu)\n",
                  now_ms() - t_bind, LhsM.rows, LhsM.values.size());

    // Synthetic SPD-projected RHS (so CG has a defined converged
    // answer in the presence of LhsM's 4-dim sample-slot null space).
    std::vector<double> y_seed(nv);
    for (std::size_t i = 0; i < nv; ++i) {
        y_seed[i] = std::sin(0.001 * static_cast<double>(i + 1));
    }
    const std::vector<double> b = sp::spmv(LhsM, y_seed);

    // ---- CPU CG reference ----
    const double t_cpu = now_ms();
    sp::CgStats cpu_stats;
    const std::vector<double> x_cpu =
        sp::cg_diag(LhsM, b, std::max<std::size_t>(50, nv * 2),
                      1e-8, cpu_stats);
    const double cpu_ms = now_ms() - t_cpu;
    {
        const std::vector<double> Ax = sp::spmv(LhsM, x_cpu);
        double max_r = 0.0;
        for (std::size_t i = 0; i < nv; ++i) {
            const double r = std::fabs(Ax[i] - b[i]);
            if (r > max_r) max_r = r;
        }
        std::printf("CPU CG: %.0f ms  iters=%zu  max_resid=%.3e\n",
                      cpu_ms, cpu_stats.iters, max_r);
    }

    // ---- GPU CG ----
    VulkanCompute vk;
    vk.init();
    GpuCgSolver gpu;
    gpu.init(vk,
              load_spv("bin/spmv.spv"),
              load_spv("bin/dot_reduce.spv"),
              load_spv("bin/axpy.spv"),
              load_spv("bin/jacobi.spv"),
              load_spv("bin/saxpby.spv"));

    const double t_prep = now_ms();
    gpu.prepare_matrix(LhsM);
    std::printf("GPU prepare_matrix: %.0f ms\n", now_ms() - t_prep);

    std::size_t gpu_iters = 0;
    const double t_gpu = now_ms();
    const std::vector<double> x_gpu =
        gpu.solve(b, std::max<std::size_t>(50, nv * 2), 1e-8, &gpu_iters);
    const double gpu_ms = now_ms() - t_gpu;
    {
        const std::vector<double> Ax = sp::spmv(LhsM, x_gpu);
        double max_r = 0.0;
        for (std::size_t i = 0; i < nv; ++i) {
            const double r = std::fabs(Ax[i] - b[i]);
            if (r > max_r) max_r = r;
        }
        std::printf("GPU CG: %.0f ms  iters=%zu  max_resid=%.3e\n",
                      gpu_ms, gpu_iters, max_r);
    }

    gpu.shutdown();
    vk.shutdown();
    return 0;
}
