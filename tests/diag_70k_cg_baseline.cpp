// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
//
// Sanity-check baseline: does plain (unpreconditioned) CG actually
// converge on the 81k cut-mesh Laplacian? If yes, the multilevel
// stall is a preconditioner issue. If no, it's a matrix issue
// (singular or extremely ill-conditioned).

#include "curvenet/cut_mesh.h"
#include "curvenet/cut_mesh_laplacian.h"
#include "curvenet/halfedge_builder.h"
#include "curvenet/sparse_linalg.h"
#include "curvenet/vec3.h"
#include "mire_body_70k_data.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using curvenet::Vec3;
namespace cm  = curvenet::cut_mesh;
namespace cml = curvenet::cut_mesh_laplacian;
namespace sp  = curvenet::sparse;

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

    const curvenet::HalfedgeMesh hm =
        curvenet::halfedge_builder::from_triangles(nv, tris);
    cm::CutMesh c;
    c.base = hm;
    c.vertex_kind.assign(nv, cm::CutVertexKind::mesh_vertex_kind());
    c.segment_of_halfedge.assign(hm.he_count(), -1);
    const double mol_delta = cml::default_mollify_delta(positions, tris);
    const sp::SparseMatrixCSR A = cml::assemble_vt_lh_v_csr_robust(
                                     c, positions, mol_delta);
    std::printf("81k: nv=%zu nnz=%zu\n", nv, A.values.size());

    // Probe row-sum spectrum: max |row_sum| tells us how far A·1 is
    // from zero (i.e. how broken the constant kernel is by Tikhonov).
    double max_row_sum = 0.0;
    double min_diag = 1e300, max_diag = 0.0;
    for (std::size_t i = 0; i < nv; ++i) {
        double s = 0.0;
        double d = 0.0;
        for (int k = A.row_ptr[i]; k < A.row_ptr[i + 1]; ++k) {
            s += A.values[k];
            if (A.col_idx[k] == static_cast<int>(i)) d = A.values[k];
        }
        if (std::fabs(s) > max_row_sum) max_row_sum = std::fabs(s);
        if (d > 0) {
            if (d < min_diag) min_diag = d;
            if (d > max_diag) max_diag = d;
        }
    }
    std::printf("max |row_sum|: %.3e  diag range: [%.3e, %.3e]\n",
                  max_row_sum, min_diag, max_diag);

    std::mt19937_64 rng(0xc0ffeeULL);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> y_seed(nv);
    for (auto &v : y_seed) v = dist(rng);
    const std::vector<double> b = sp::spmv(A, y_seed);

    // y_seed mean and b mean.
    double y_sum = 0.0; for (double x : y_seed) y_sum += x;
    double b_sum = 0.0; for (double x : b)      b_sum += x;
    std::printf("y_seed mean = %.3e   b mean = %.3e\n",
                  y_sum / nv, b_sum / nv);

    // Plain CG with diagnostic stats.
    using namespace std::chrono;
    const auto t0 = steady_clock::now();
    sp::CgStats stats;
    sp::cg_diag(A, b, 200000, 1e-18, stats);
    const auto t1 = steady_clock::now();
    std::printf("plain CG: %zu iters, final_rr = %.3e, %.2f s\n",
                  stats.iters, stats.final_rr,
                  duration_cast<duration<double>>(t1 - t0).count());
    return 0;
}
