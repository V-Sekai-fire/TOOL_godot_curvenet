// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
//
// Single-solve bench at the 70k-vert PCVR target. Loads the
// 2x-subdivided Mire body (81,613 verts, ~159k tris) baked into
// `tests/mire_body_70k_data.h`, runs the existing monolithic
// sparse-CSR + Jacobi-PCG + warm-start path that the current
// runtime uses, and prints bind / per-frame numbers — no
// extrapolation.
//
// Why a separate exec: the 70k header is 6.4 MB and bloats
// every compile that touches mire_body_data.h. Keeping it
// isolated saves rebuild time on the smaller benches.
//
// Per Gall's Law: the goal is to MEASURE what the working
// simple system actually does at the deployment target before
// designing meshlet replacements that may or may not be needed.

#include "curvenet/cut_mesh.h"
#include "curvenet/cut_mesh_laplacian.h"
#include "curvenet/halfedge_builder.h"
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
namespace sp  = curvenet::sparse;

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

    std::printf("Mire body 70k: %d verts, %d tris\n",
                  mire_body_70k::n_verts, mire_body_70k::n_tris);

    // ---- Load + halfedge ----
    const double t_load = now_ms();
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
    const std::size_t nh = hm.he_count();
    std::printf("load+halfedge: %.2f ms  (nh = %zu)\n",
                  now_ms() - t_load, nh);

    // ---- Cut mesh + 4 sample-promoted verts ----
    cm::CutMesh c;
    c.base = hm;
    c.vertex_kind.assign(nv, cm::CutVertexKind::mesh_vertex_kind());
    c.segment_of_halfedge.assign(hm.he_count(), -1);
    const std::vector<Vec3> sample_targets = {
        {  0.4,  0.0, 1.0 },   // right waist
        { -0.4,  0.0, 1.0 },   // left  waist
        {  0.0,  0.0, 1.6 },   // chest
        {  0.0,  0.0, 0.6 },   // pelvis
    };
    const int nc = static_cast<int>(sample_targets.size());
    for (std::size_t s = 0; s < sample_targets.size(); ++s) {
        const int vi = closest_vertex(positions, sample_targets[s]);
        c.vertex_kind[static_cast<std::size_t>(vi)] =
            cm::CutVertexKind::sample_kind(static_cast<int>(s), 0, false);
    }

    // ---- Bind: assemble Lh + LhsM (robust mollified Laplacian) ----
    const double t_bind = now_ms();
    const double mol_delta = cml::default_mollify_delta(positions, tris);
    const sp::SparseMatrixCSR Lh_csr =
        cml::assemble_lh_csr_robust(c, positions, mol_delta);
    const sp::SparseMatrixCSR LhsM_csr =
        cml::assemble_vt_lh_v_csr_robust(c, positions, mol_delta);
    const double bind_ms = now_ms() - t_bind;
    std::printf("bind  (Lh + LhsM assemble): %.2f ms\n", bind_ms);
    std::printf("  Lh  : rows=%zu nnz=%zu\n", Lh_csr.rows, Lh_csr.values.size());
    std::printf("  LhsM: rows=%zu nnz=%zu\n", LhsM_csr.rows, LhsM_csr.values.size());

    // ---- Single per-frame solve, no warm-start, just so we have a
    //      hard cold-start number. ----
    // Build a synthetic SPD-projected RHS so CG has a well-defined
    // converged answer on the singular LhsM (sample slots have zero
    // rows; Jacobi-PCG handles them by setting y[i] = 0).
    std::vector<double> y_seed(nv);
    for (std::size_t i = 0; i < nv; ++i) {
        y_seed[i] = std::sin(0.001 * static_cast<double>(i + 1));
    }
    const std::vector<double> rhs = sp::spmv(LhsM_csr, y_seed);

    // 12 RHS columns (Fv 9 + Xv 3) on the same LHS, mimicking the
    // deformer's per-frame load.
    const std::size_t k = 12;
    const std::size_t cg_max_iter = std::max<std::size_t>(50, nv * 2);
    const double tol = 1e-8;

    const double t_solve = now_ms();
    for (std::size_t col = 0; col < k; ++col) {
        const std::vector<double> x = sp::cg(LhsM_csr, rhs, cg_max_iter, tol);
        (void)x;
    }
    const double solve_ms = now_ms() - t_solve;

    // Residual on the last-column solve, just so we know convergence.
    const std::vector<double> x_last = sp::cg(LhsM_csr, rhs, cg_max_iter, tol);
    const std::vector<double> Ax = sp::spmv(LhsM_csr, x_last);
    double max_resid = 0.0;
    for (std::size_t i = 0; i < nv; ++i) {
        const double r = std::fabs(Ax[i] - rhs[i]);
        if (r > max_resid) max_resid = r;
    }

    std::printf("solve (12 RHS cold-start CGs): %.2f ms  (~%.2f ms / RHS)\n",
                  solve_ms, solve_ms / static_cast<double>(k));
    std::printf("last RHS max_resid: %.3e\n", max_resid);

    std::printf("\n--- summary ---\n");
    std::printf("nv=%zu nh=%zu nc=%d\n", nv, nh, nc);
    std::printf("bind: %.0f ms  solve: %.0f ms  → frames/s: %.2f\n",
                  bind_ms, solve_ms, 1000.0 / solve_ms);
    std::printf("note: cold-start; warm-start typically halves iter count\n");
    return 0;
}
