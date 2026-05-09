// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
//
// ICC(0)-preconditioned CG on the 5k Mire cut-mesh Laplacian.
//
// Test gate (set in PERF_BASELINE.md "Live candidate"): drop a
// candidate smoother into a 1-level diag and confirm it converges
// in isolation. Plain CG baseline (diag_70k_cg_baseline.cpp):
// tbd iters / tbd s -> ‖r‖² = 3.6e-10. Goal: ICC(0) cuts that
// to under 1000 iters / a few seconds.

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
#include <random>
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

// ICC(0)-preconditioned CG. Mirror of `sp::cg` with the inner
// `apply_jacobi` swapped for ICC backsolve.
struct PcgStats {
    std::size_t iters;
    double      final_rr;
    bool        converged;
};

PcgStats cg_icc(const sp::SparseMatrixCSR &A,
                const icc::IncompleteCholeskyFactor &fac,
                const std::vector<double> &b,
                std::vector<double> &x_out,
                std::size_t max_iter,
                double tol) {
    const std::size_t n = A.rows;
    std::vector<double> x(n, 0.0);
    std::vector<double> r = b;
    std::vector<double> z = icc::apply_minv(fac, r);
    std::vector<double> p = z;
    double rz_old = sp::dot(r, z);
    const double tol_sq = tol * tol;
    PcgStats st { 0, sp::dot(r, r), false };
    for (std::size_t iter = 0; iter < max_iter; ++iter) {
        const std::vector<double> Ap = sp::spmv(A, p);
        const double pAp = sp::dot(p, Ap);
        if (pAp == 0.0) { st.iters = iter; break; }
        const double alpha = rz_old / pAp;
        sp::axpy_inplace(alpha, p, x);
        sp::axpy_inplace(-alpha, Ap, r);
        st.final_rr = sp::dot(r, r);
        st.iters = iter + 1;
        if (iter < 5 || iter % 10 == 9 || st.final_rr < tol_sq) {
            std::printf("  iter %6zu  ‖r‖² = %.4e\n", iter, st.final_rr);
        }
        if (st.final_rr < tol_sq) { st.converged = true; break; }
        z = icc::apply_minv(fac, r);
        const double rz_new = sp::dot(r, z);
        const double beta = (rz_old == 0.0) ? 0.0 : (rz_new / rz_old);
        p = sp::saxpby(1.0, z, beta, p);
        rz_old = rz_new;
    }
    x_out = std::move(x);
    return st;
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

    // Factorise. Retry with progressively larger diagonal shift if
    // the no-fill variant breaks down (Manteuffel 1980 shifted-ICC).
    const double t_fact_start = now_ms();
    icc::IncompleteCholeskyFactor fac;
    double used_shift = 0.0;
    for (double s : { 0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0 }) {
        fac = icc::factor(A, s);
        used_shift = s;
        if (!fac.breakdown) break;
    }
    const double t_fact = now_ms() - t_fact_start;
    if (fac.breakdown) {
        std::printf("ICC(0) BREAKDOWN even at shift = 1.0\n");
        return 2;
    }
    if (used_shift > 0.0) {
        std::printf("ICC(0) needed diagonal shift = %.0e\n", used_shift);
    }
    std::printf("ICC(0) factorisation: nnz(L) = %zu, %.2f ms\n",
                  fac.L.values.size(), t_fact);

    // RHS.
    std::mt19937_64 rng(0xc0ffeeULL);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> y_seed(nv);
    for (auto &v : y_seed) v = dist(rng);
    const std::vector<double> b = sp::spmv(A, y_seed);

    // Solve.
    std::vector<double> x;
    const double t_solve_start = now_ms();
    std::printf("\nICC(0)-PCG (max 90s):\n");
    const PcgStats st = cg_icc(A, fac, b, x, 100000, 1e-9);
    const double t_solve = now_ms() - t_solve_start;

    if (st.converged) {
        std::printf("\nICC(0)-PCG: converged in %zu iters / %.2f s\n",
                      st.iters, t_solve / 1000.0);
        std::printf("vs plain CG baseline: tbd iters / tbd s\n");
        if (st.iters > 0) {
            std::printf("speedup: %.0fx in iter count, %.0fx in wall time\n",
                          0.0 / st.iters, 0.0 / t_solve);
        }
    } else {
        std::printf("\nICC(0)-PCG: did NOT converge in %zu iters / %.2f s\n"
                      "  final ‖r‖² = %.3e\n",
                      st.iters, t_solve / 1000.0, st.final_rr);
    }
    return st.converged ? 0 : 1;
}
