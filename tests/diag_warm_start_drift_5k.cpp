// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
//
// Reproduce the loop-100/5 warm-start divergence on Mire 5k and
// dump per-iter (r_inf, x_inf, alpha, beta) so the failure mode
// can be diagnosed directly. Hard 5 s wall cap.
//
// Recipe:
//   1. cold solve to get x_warm (converges)
//   2. perturb b -> b_pert, project zero-mean (range(A))
//   3. warm-start solve from x_warm; print per-iter trace
//   4. compare to cold solve from x0 = 0 on the same b_pert

#include "curvenet/cut_mesh.h"
#include "curvenet/cut_mesh_laplacian.h"
#include "curvenet/halfedge_builder.h"
#include "curvenet/incomplete_cholesky.h"
#include "curvenet/sparse_linalg.h"
#include "curvenet/vec3.h"
#include "mire_body_data.h"

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

void print_trace(const char *tag,
                    const std::vector<icc::PcgIterTrace> &t,
                    std::size_t every) {
    std::fflush(stdout);
    std::printf("== %s ==\n", tag);
    std::printf("%-5s  %-11s  %-11s  %-11s  %-11s  %-11s\n",
                  "iter", "r_inf", "x_inf", "alpha", "beta", "rz");
    for (std::size_t i = 0; i < t.size(); ++i) {
        if (i < 5 || i % every == 0 || i + 1 == t.size()) {
            std::printf("%-5zu  %.4e   %.4e   %+.4e  %+.4e  %.4e\n",
                          t[i].iter, t[i].r_inf, t[i].x_inf,
                          t[i].alpha, t[i].beta, t[i].rz);
        }
    }
    std::fflush(stdout);
}

} // namespace

int main() {
    const double T_CAP_MS = 5000.0;
    const double t_start = now_ms();

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
    const double mol_delta = cml::default_mollify_delta(positions, tris);
    const sp::SparseMatrixCSR A = cml::assemble_vt_lh_v_csr_robust(
                                     c, positions, mol_delta);
    std::printf("5k: nv=%zu nnz=%zu\n", nv, A.values.size());

    double shift = 0.0;
    const icc::IncompleteCholeskyFactor fac =
        icc::factor_with_retry(A, &shift);
    if (fac.breakdown) { std::printf("ICC breakdown\n"); return 2; }
    std::printf("ICC factor shift = %.0e, nnz(L) = %zu\n",
                  shift, fac.L.values.size());

    // Right-hand side b = A · y_seed (zero-mean by construction since
    // y_seed[i] = sin(...) which sums to ~0 and A·1 ≈ 0).
    std::vector<double> y_seed(nv);
    for (std::size_t i = 0; i < nv; ++i) {
        y_seed[i] = std::sin(0.001 * static_cast<double>(i + 1));
    }
    const std::vector<double> b = sp::spmv(A, y_seed);

    // Cap iters tight so divergence shows up fast.
    const std::size_t max_iter = 500;
    const double tol = 1e-9;

    // ---- A. cold solve, dump trace ----
    {
        std::vector<icc::PcgIterTrace> trace;
        const std::vector<double> x0(nv, 0.0);
        const std::vector<double> x = icc::cg_icc_with_guess_diag(
            A, fac, b, x0, max_iter, tol, trace);
        print_trace("cold-start solve (x0 = 0)", trace, 50);
        // Sanity: residual at end.
        const std::vector<double> Ax = sp::spmv(A, x);
        double r_inf = 0.0;
        for (std::size_t i = 0; i < nv; ++i) {
            r_inf = std::max(r_inf, std::fabs(Ax[i] - b[i]));
        }
        std::printf("cold final |r|_inf = %.3e\n\n", r_inf);
    }

    if (now_ms() - t_start > T_CAP_MS) {
        std::printf("WALL-CAP HIT after cold solve\n");
        return 0;
    }

    // ---- B. warm solve same b: should bail in 0-1 iter ----
    // First produce x_warm via a normal cold solve.
    std::vector<double> x_warm;
    {
        const std::vector<double> x0(nv, 0.0);
        x_warm = icc::cg_icc_with_guess(A, fac, b, x0, max_iter, tol);
    }
    {
        std::vector<icc::PcgIterTrace> trace;
        icc::cg_icc_with_guess_diag(A, fac, b, x_warm,
                                          max_iter, tol, trace);
        print_trace("warm same-b (x0 = x_warm)", trace, 1);
    }

    if (now_ms() - t_start > T_CAP_MS) return 0;

    // ---- C. warm solve, REALISTIC drifted b ----
    // Real frame-to-frame perturbation in the deformer is
    //   b' = b + A·delta_x
    // where delta_x is a small smooth shift of the rest pose.
    // This is automatically in range(A) and the new solution
    // is x_warm + delta_x — bounded magnitude. Synthetic
    // sin(...) perturbations hit A's low-frequency modes hard
    // and require enormous true solutions fp64 can't track —
    // so they look like "warm-start broken" but are really
    // just an unrealistic test signal.
    std::vector<double> delta_x(nv);
    for (std::size_t i = 0; i < nv; ++i) {
        delta_x[i] = 1e-2 * std::sin(0.0011 * static_cast<double>(i));
    }
    std::vector<double> y_seed_pert(nv);
    for (std::size_t i = 0; i < nv; ++i) {
        y_seed_pert[i] = y_seed[i] + delta_x[i];
    }
    const std::vector<double> b_pert = sp::spmv(A, y_seed_pert);

    // Dump pre-iter state: initial residual + initial search direction.
    {
        const std::vector<double> Ax_warm = sp::spmv(A, x_warm);
        std::vector<double> r0(nv);
        for (std::size_t i = 0; i < nv; ++i) r0[i] = b_pert[i] - Ax_warm[i];
        const std::vector<double> z0 = icc::apply_minv(fac, r0);
        double r0_inf = 0.0, z0_inf = 0.0, x_warm_inf = 0.0;
        for (double v : r0)     r0_inf = std::max(r0_inf, std::fabs(v));
        for (double v : z0)     z0_inf = std::max(z0_inf, std::fabs(v));
        for (double v : x_warm) x_warm_inf = std::max(x_warm_inf, std::fabs(v));
        std::printf("\n-- pre-iter state for drifted-b warm solve --\n");
        std::printf("  |x_warm|_inf  = %.4e\n", x_warm_inf);
        std::printf("  |r0|_inf      = %.4e   (= |b_pert - A x_warm|_inf)\n", r0_inf);
        std::printf("  |M^{-1} r0|_inf = %.4e   (initial search dir)\n", z0_inf);
        std::printf("  amplification |z0|/|r0| = %.4e\n", z0_inf / r0_inf);
        std::fflush(stdout);
    }

    {
        std::vector<icc::PcgIterTrace> trace;
        icc::cg_icc_with_guess_diag(A, fac, b_pert, x_warm,
                                          max_iter, tol, trace,
                                          /*project_kernel=*/false);
        print_trace("warm drifted-b NO PROJECTION (diverges)", trace, 50);
    }
    // Sanity: COLD solve on the same b_pert. If this converges,
    // the divergence is warm-start specific. If it also diverges,
    // the matrix factor is the problem.
    {
        std::vector<icc::PcgIterTrace> trace;
        const std::vector<double> x0(nv, 0.0);
        icc::cg_icc_with_guess_diag(A, fac, b_pert, x0,
                                          max_iter, tol, trace,
                                          /*project_kernel=*/false);
        print_trace("COLD on b_pert (control)", trace, 50);
    }
    {
        std::vector<icc::PcgIterTrace> trace;
        const std::vector<double> x = icc::cg_icc_with_guess_diag(
            A, fac, b_pert, x_warm, max_iter, tol, trace,
            /*project_kernel=*/true);
        print_trace("warm drifted-b WITH PROJECTION (loop 100/6 fix)",
                       trace, 5);
        const std::vector<double> Ax = sp::spmv(A, x);
        double r_inf = 0.0;
        for (std::size_t i = 0; i < nv; ++i) {
            r_inf = std::max(r_inf, std::fabs(Ax[i] - b_pert[i]));
        }
        std::printf("warm drifted-b WITH PROJECTION final |r|_inf = %.3e\n",
                      r_inf);
    }

    std::printf("\n(total bench: %.1f s, cap %.0f s)\n",
                  (now_ms() - t_start) / 1000.0, T_CAP_MS / 1000.0);
    return 0;
}
