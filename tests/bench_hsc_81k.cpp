// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
//
// HSC cycle 6: bench HSC-PCG vs ICC(0)-PCG vs plain D-Jacobi PCG
// on the Mire 81k cot-Laplacian. Hard 5 s wall cap.
//
// If HSC is doing what the paper claims, kappa(M^-1 * A) should
// be O(1) - we'd see PCG converge in dozens of iters at most,
// vs ICC's ~350 iters. That's the gating measurement before
// committing to HSC as the production solver path.

#include "curvenet/cut_mesh.h"
#include "curvenet/cut_mesh_laplacian.h"
#include "curvenet/halfedge_builder.h"
#include "curvenet/hierarchical_sparsify.h"
#include "curvenet/incomplete_cholesky.h"
#include "curvenet/sparse_linalg.h"
#include "curvenet/vec3.h"
#include "mire_body_70k_data.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

using curvenet::Vec3;
namespace cm   = curvenet::cut_mesh;
namespace cml  = curvenet::cut_mesh_laplacian;
namespace icc  = curvenet::incomplete_cholesky;
namespace hscn = curvenet::hsc;
namespace sp   = curvenet::sparse;

namespace {

double now_ms() {
    using namespace std::chrono;
    return duration_cast<duration<double, std::milli>>(
        steady_clock::now().time_since_epoch()).count();
}

} // namespace

int main() {
    // Generous cap for the gating measurement: HSC build alone is
    // ~29 s at 81k without sparsify; we want PCG iter count + per-
    // iter ms before commitment to AMG-classical rewrite.
    const double T_CAP_MS = 90000.0;
    const double t_start = now_ms();

    std::vector<Vec3> positions;
    positions.reserve(mire_body_70k::n_verts);
    for (int i = 0; i < mire_body_70k::n_verts; ++i) {
        positions.push_back({
            static_cast<double>(mire_body_70k::positions[i * 3 + 0]),
            static_cast<double>(mire_body_70k::positions[i * 3 + 1]),
            static_cast<double>(mire_body_70k::positions[i * 3 + 2]),
        });
    }
    std::vector<int> tris(mire_body_70k::tris,
                            mire_body_70k::tris + mire_body_70k::n_tris * 3);
    const std::size_t nv = positions.size();

    const curvenet::HalfedgeMesh hm =
        curvenet::halfedge_builder::from_triangles(nv, tris);
    cm::CutMesh c;
    c.base = hm;
    c.vertex_kind.assign(nv, cm::CutVertexKind::mesh_vertex_kind());
    c.segment_of_halfedge.assign(hm.he_count(), -1);
    const double mol_delta = cml::default_mollify_delta(positions, tris);
    const sp::SparseMatrixCSR A = cml::assemble_vt_lh_v_csr_robust(
                                     c, positions, mol_delta);
    std::printf("Mire 81k: nv=%zu, nnz=%zu\n", nv, A.values.size());
    std::fflush(stdout);

    // Build HSC hierarchy.
    const double t_hb = now_ms();
    const hscn::Graph g = hscn::csr_to_graph(A);
    const hscn::Hierarchy h = hscn::build_hierarchy(g, /*coarsest_size=*/64,
                                                            /*max_levels=*/64,
                                                            /*sparsify_tau=*/0.0,
                                                            /*max_degree=*/32,
                                                            /*verbose=*/false);
    const double hsc_build_ms = now_ms() - t_hb;
    std::printf("HSC hierarchy: %zu levels, sizes ", h.graphs.size());
    for (const auto &gr : h.graphs) std::printf("%zu ", gr.num_verts);
    std::printf("[build %.1f ms]\n", hsc_build_ms);
    std::fflush(stdout);

    if (now_ms() - t_start > T_CAP_MS) {
        std::printf("WALL-CAP HIT after hierarchy build\n"); return 2;
    }

    // Skip ICC factor at 81k — known to take 28 s; HSC is what
    // we want to measure here. Reuse the dummy factor below for
    // the (void)fac; suppression.
    icc::IncompleteCholeskyFactor fac;

    // RHS: b = A · y_seed, in range(A).
    std::vector<double> y_seed(nv);
    for (std::size_t i = 0; i < nv; ++i) {
        y_seed[i] = std::sin(0.001 * static_cast<double>(i + 1));
    }
    const std::vector<double> b = sp::spmv(A, y_seed);
    const std::vector<double> x0(nv, 0.0);

    auto measure = [&](const char *tag, auto solver) {
        const double t = now_ms();
        const std::vector<double> x = solver();
        const double ms = now_ms() - t;
        const std::vector<double> Ax = sp::spmv(A, x);
        double r2 = 0.0;
        for (std::size_t i = 0; i < nv; ++i) {
            const double r = Ax[i] - b[i];
            r2 += r * r;
        }
        std::printf("  %-22s : %8.2f ms   ||r|| = %.3e\n",
                      tag, ms, std::sqrt(r2));
        std::fflush(stdout);
    };

    // HSC is the load-bearing measurement at 81k — D-Jacobi and
    // ICC are documented as 13.5 s and 3.5 s/RHS from prior runs;
    // we want HSC's per-RHS time, not to spend 30+ s re-measuring.
    std::printf("\n--- single-RHS HSC solve at tol=1e-8 ---\n");
    {
        std::size_t hsc_iters = 0;
        const double t = now_ms();
        const std::vector<double> x = hscn::cg_hsc_with_guess(
            A, h, b, x0, 1000, 1e-8, &hsc_iters);
        const double ms = now_ms() - t;
        const std::vector<double> Ax = sp::spmv(A, x);
        double r2 = 0.0;
        for (std::size_t i = 0; i < nv; ++i) {
            const double r = Ax[i] - b[i];
            r2 += r * r;
        }
        std::printf("  %-22s : %8.2f ms   ||r|| = %.3e   iters=%zu\n",
                      "HSC-PCG (V-cycle)", ms, std::sqrt(r2), hsc_iters);
        std::fflush(stdout);
        std::printf("\n--- comparison (from prior bench_deform_70k_icc) ---\n");
        std::printf("  D-Jacobi PCG cold      : 13500 ms (recorded earlier)\n");
        std::printf("  ICC(0)-PCG cold        :  3487 ms (recorded earlier)\n");
        std::printf("  HSC-PCG cold (this run): %.0f ms  -> %.1fx vs ICC\n",
                      ms, 3487.0 / ms);
    }
    (void)fac;   // ICC factor still useful as control; suppress unused warn

    std::printf("(total: %.1f s, cap %.0f s)\n",
                  (now_ms() - t_start) / 1000.0, T_CAP_MS / 1000.0);
    return 0;
}
