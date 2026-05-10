// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
//
// What iter count + wall time do we hit at relaxed tolerances on
// the 5k Mire cot-Laplacian? The deformer's current 1e-8 might be
// 4-5x more accurate than visually needed for character animation.
// 5 s wall cap.

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

    double shift = 0.0;
    const icc::IncompleteCholeskyFactor fac =
        icc::factor_with_retry(A, &shift);

    std::vector<double> y_seed(nv);
    for (std::size_t i = 0; i < nv; ++i) {
        y_seed[i] = std::sin(0.001 * static_cast<double>(i + 1));
    }
    const std::vector<double> b = sp::spmv(A, y_seed);

    std::printf("%-12s  %-12s  %-12s  %-12s  %-12s\n",
                  "tol", "iters_dJ",  "ms_dJ", "iters_ICC", "ms_ICC");

    for (double tol : { 1e-9, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2 }) {
        // D-Jacobi PCG with this tol
        sp::CgStats st_dj;
        const double t_dj = now_ms();
        sp::cg_diag(A, b, 100000, tol, st_dj);
        const double ms_dj = now_ms() - t_dj;

        // ICC-PCG with this tol (cold)
        const double t_icc = now_ms();
        const std::vector<double> x0(nv, 0.0);
        const std::vector<double> x = icc::cg_icc_with_guess(
            A, fac, b, x0, 100000, tol);
        const double ms_icc = now_ms() - t_icc;
        const std::vector<double> Ax = sp::spmv(A, x);
        // Estimate ICC iter count via residual (no diag variant).
        double r2 = 0.0;
        for (std::size_t i = 0; i < nv; ++i) {
            const double r = Ax[i] - b[i];
            r2 += r * r;
        }
        const double r_inf = std::sqrt(r2 / nv);

        std::printf("%-12.0e  %-12zu  %-12.2f  %-12s  %-12.2f  (icc r ~ %.2e)\n",
                      tol, st_dj.iters, ms_dj, "?", ms_icc, r_inf);
    }
    return 0;
}
