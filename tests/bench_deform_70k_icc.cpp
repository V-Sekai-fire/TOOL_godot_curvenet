// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
//
// 81k bench using ICC(0)-PCG instead of D-Jacobi PCG. Same workload
// shape as bench_deform_70k.cpp (12 RHS on the cached LhsM_csr,
// no warm-start) so the two are directly comparable.
//
// Per loop 100/4: this is the next Gall's-law step after
// diag_70k_icc_pcg. The diagnostic showed ICC-PCG converges in
// 2,478 iters / 3.65 s on a single RHS. This bench confirms the
// 12-RHS deformer-shaped workload and the one-shot factorisation
// cost amortised across them.

#include "curvenet/cut_mesh.h"
#include "curvenet/cut_mesh_laplacian.h"
#include "curvenet/halfedge_builder.h"
#include "curvenet/incomplete_cholesky.h"
#include "curvenet/sparse_linalg.h"
#include "curvenet/vec3.h"
#include "mire_body_70k_data.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

using curvenet::Vec3;
namespace cm  = curvenet::cut_mesh;
namespace cml = curvenet::cut_mesh_laplacian;
namespace icc = curvenet::incomplete_cholesky;
namespace sp  = curvenet::sparse;

namespace {

double now_ms() {
    using namespace std::chrono;
    return duration_cast<duration<double, std::milli>>(
        steady_clock::now().time_since_epoch()).count();
}

} // namespace

int main() {
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
    std::printf("Mire body 81k: %zu verts, %zu tris\n",
                  nv, tris.size() / 3);

    const curvenet::HalfedgeMesh hm =
        curvenet::halfedge_builder::from_triangles(nv, tris);
    cm::CutMesh c;
    c.base = hm;
    c.vertex_kind.assign(nv, cm::CutVertexKind::mesh_vertex_kind());
    c.segment_of_halfedge.assign(hm.he_count(), -1);
    const double mol_delta = cml::default_mollify_delta(positions, tris);
    const double t_assem = now_ms();
    const sp::SparseMatrixCSR LhsM_csr =
        cml::assemble_vt_lh_v_csr_robust(c, positions, mol_delta);
    const double assem_ms = now_ms() - t_assem;
    std::printf("LhsM assemble: %.2f ms  (rows=%zu nnz=%zu)\n",
                  assem_ms, LhsM_csr.rows, LhsM_csr.values.size());

    // Bind-time ICC factorisation. Amortised across all subsequent
    // solves; for a single rest-pose this is paid once, then every
    // frame just runs cg_icc_with_guess.
    const double t_fact = now_ms();
    double shift = 0.0;
    const icc::IncompleteCholeskyFactor fac = icc::factor_with_retry(LhsM_csr, &shift);
    const double fact_ms = now_ms() - t_fact;
    if (fac.breakdown) {
        std::printf("ICC(0) BREAKDOWN even at shift=1.0\n");
        return 1;
    }
    std::printf("ICC(0) factor: %.2f ms  (nnz(L)=%zu, shift=%.0e)\n",
                  fact_ms, fac.L.values.size(), shift);
    const double bind_ms = assem_ms + fact_ms;
    std::printf("bind total (assemble + factor): %.2f ms\n", bind_ms);

    // Synthetic RHS, same recipe as bench_deform_70k.
    std::vector<double> y_seed(nv);
    for (std::size_t i = 0; i < nv; ++i) {
        y_seed[i] = std::sin(0.001 * static_cast<double>(i + 1));
    }
    const std::vector<double> rhs = sp::spmv(LhsM_csr, y_seed);

    // 12 RHS columns (Fv 9 + Xv 3) on the same LHS.
    const std::size_t k = 12;
    const std::size_t cg_max_iter = std::max<std::size_t>(50, nv * 2);
    const double tol = 1e-8;

    const std::vector<double> x0(nv, 0.0);
    const double t_solve = now_ms();
    std::vector<double> x_last;
    for (std::size_t col = 0; col < k; ++col) {
        x_last = icc::cg_icc_with_guess(LhsM_csr, fac, rhs, x0,
                                            cg_max_iter, tol);
    }
    const double solve_ms = now_ms() - t_solve;

    // Final residual.
    const std::vector<double> Ax = sp::spmv(LhsM_csr, x_last);
    double max_resid = 0.0;
    for (std::size_t i = 0; i < nv; ++i) {
        const double r = std::fabs(Ax[i] - rhs[i]);
        if (r > max_resid) max_resid = r;
    }
    const double per_rhs_ms = solve_ms / static_cast<double>(k);

    std::printf("solve (12 RHS cold-start ICC-PCGs): %.2f ms  (~%.2f ms / RHS)\n",
                  solve_ms, per_rhs_ms);
    std::printf("last RHS max_resid: %.3e\n", max_resid);

    std::printf("\n--- summary ---\n");
    std::printf("nv=%zu\n", nv);
    std::printf("bind: %.0f ms  solve: %.0f ms  -> frames/s: %.2f\n",
                  bind_ms, solve_ms, 1000.0 / solve_ms);
    std::printf("vs D-Jacobi PCG: bind 172 ms, solve 162053 ms -> %.0fx wall\n",
                  162053.0 / solve_ms);
    return 0;
}
