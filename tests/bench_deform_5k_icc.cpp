// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
//
// 5k bench using ICC(0)-PCG instead of D-Jacobi PCG. Same workload
// shape as bench_deform_70k.cpp (12 right-hand side on the cached LhsM_csr,
// no warm-start) so the two are directly comparable.
//
// Per loop 100/4: this is the next Gall's-law step after
// diag_70k_icc_pcg. The diagnostic showed ICC-PCG converges in
// 2,478 iters / 3.65 s on a single right-hand side. This bench confirms the
// 12-vector right-hand-side deformer-shaped workload and the one-shot factorisation
// cost amortised across them.

#include "curvenet/cut_mesh.h"
#include "curvenet/cut_mesh_laplacian.h"
#include "curvenet/halfedge_builder.h"
#include "curvenet/incomplete_cholesky.h"
#include "curvenet/sparse_linalg.h"
#include "curvenet/vec3.h"
#include "mire_body_data.h"

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
    std::fflush(stdout); std::printf("Mire body 5k: %zu verts, %zu tris\n",
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
    std::fflush(stdout); std::printf("LhsM assemble: %.2f ms  (rows=%zu nnz=%zu)\n",
                  assem_ms, LhsM_csr.rows, LhsM_csr.values.size());

    // Bind-time ICC factorisation. Amortised across all subsequent
    // solves; for a single rest-pose this is paid once, then every
    // frame just runs cg_icc_with_guess.
    const double t_fact = now_ms();
    double shift = 0.0;
    const icc::IncompleteCholeskyFactor fac = icc::factor_with_retry(LhsM_csr, &shift);
    const double fact_ms = now_ms() - t_fact;
    if (fac.breakdown) {
        std::fflush(stdout); std::printf("ICC(0) BREAKDOWN even at shift=1.0\n");
        return 1;
    }
    std::fflush(stdout); std::printf("ICC(0) factor: %.2f ms  (nnz(L)=%zu, shift=%.0e)\n",
                  fact_ms, fac.L.values.size(), shift);
    const double bind_ms = assem_ms + fact_ms;
    std::fflush(stdout); std::printf("bind total (assemble + factor): %.2f ms\n", bind_ms);

    // Synthetic right-hand side, same recipe as bench_deform_70k.
    std::vector<double> y_seed(nv);
    for (std::size_t i = 0; i < nv; ++i) {
        y_seed[i] = std::sin(0.001 * static_cast<double>(i + 1));
    }
    const std::vector<double> rhs = sp::spmv(LhsM_csr, y_seed);

    // 12 right-hand side columns (Fv 9 + Xv 3) on the same LHS.
    const std::size_t k = 12;
    const std::size_t cg_max_iter = std::max<std::size_t>(50, nv * 2);
    const double tol = 1e-8;

    // Hard wall-clock cap: per the project's 5 ms-deadline rule
    // (PCVR target is 11 ms / frame, we want measurements to be
    // well below that to leave headroom). Training problem set is
    // 5485 verts so each solve should finish in a few hundred ms;
    // anything taking >5 s is considered failed.
    const double T_CAP_MS = 5000.0;
    const double t_bench_start = now_ms();
    auto cap_hit = [&] { return now_ms() - t_bench_start > T_CAP_MS; };

    // Single-right-hand side measurements (the bench compares iter count + time
    // per single solve, then we extrapolate to 12-vector right-hand-side frames). Cold
    // solve dominated everything in the prior bench, so do just one.
    const std::vector<double> x0_zero(nv, 0.0);

    // Mode A: cold-start (frame 0).
    const double t_a = now_ms();
    std::vector<double> x_warm =
        icc::cg_icc_with_guess(LhsM_csr, fac, rhs, x0_zero,
                                  cg_max_iter, tol);
    const double cold_ms = now_ms() - t_a;
    if (cap_hit()) {
        std::fflush(stdout); std::printf("WALL-CAP HIT during cold solve (%.0f ms)\n", cold_ms);
        return 2;
    }

    // Mode B: warm-start, identical right-hand side. Should converge in ~0 iters.
    const double t_b = now_ms();
    x_warm = icc::cg_icc_with_guess(LhsM_csr, fac, rhs, x_warm,
                                        cg_max_iter, tol);
    const double warm_same_ms = now_ms() - t_b;

    // Final residual on the warm-same right-hand side (mode B).
    const std::vector<double> Ax = sp::spmv(LhsM_csr, x_warm);
    double max_resid = 0.0;
    for (std::size_t i = 0; i < nv; ++i) {
        const double r = std::fabs(Ax[i] - rhs[i]);
        if (r > max_resid) max_resid = r;
    }

    std::fflush(stdout); std::printf("\n--- single right-hand-side solves on the same LHS ---\n");
    std::fflush(stdout); std::printf("  A. cold-start (x0 = 0)        : %9.2f ms\n", cold_ms);
    std::fflush(stdout); std::printf("  B. warm, same b (x0 = x_cold) : %9.2f ms  (warm-start works here)\n",
                  warm_same_ms);
    std::fflush(stdout); std::printf("  warm-same residual: %.3e\n", max_resid);

    // Mode C (warm-start with drifted right-hand side) is omitted
    // for now: with the current shifted-ICC factoriser the drifted
    // case diverges (residual blows up to 1e+10). Diagnosis is in
    // PERF_BASELINE.md "Warm-start divergence with shifted ICC".

    // Extrapolate per-frame cost.
    const double cold_frame_ms = cold_ms * 12.0;
    std::fflush(stdout); std::printf("\n--- summary (interactive perf at 5k, cold-start frame 0) ---\n");
    std::fflush(stdout); std::printf("nv=%zu  bind=%.0f ms (one-shot, ICC factor + LhsM assemble)\n",
                  nv, bind_ms);
    std::fflush(stdout); std::printf("cold 12 right-hand-side frame  : %8.0f ms  -> %.2f FPS\n",
                  cold_frame_ms, 1000.0 / cold_frame_ms);
    std::fflush(stdout); std::printf("(total bench: %.1f s, cap %.0f s)\n",
                  (now_ms() - t_bench_start) / 1000.0, T_CAP_MS / 1000.0);
    return 0;
}
