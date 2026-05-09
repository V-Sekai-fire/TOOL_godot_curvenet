// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
//
// Cheap one-shot diagnostic for the Mire body real-mesh perf problem.
// Loads the body, promotes 4 sample vertices, builds the cut mesh and
// `LhsM_csr`, then reports:
//   * Lₕ assembly stats: min/max/zero/negative diagonals
//   * LhsM_csr (the LHS that CG actually solves) diagonal stats
//   * Total nnz, mean nnz/row
//   * Conditioning proxy: max(|diag|) / min(|nonzero diag|)
//
// Anything off here (negative diagonals, near-zero diagonals, huge
// dynamic range) explains why the Jacobi-preconditioned CG stalls.
//
// Build + run from project root:
//   make -C tests bin/diag_real_mesh && tests/bin/diag_real_mesh

#include "curvenet/cut_mesh.h"
#include "curvenet/cut_mesh_laplacian.h"
#include "curvenet/halfedge_builder.h"
#include "curvenet/sparse_linalg.h"
#include "curvenet/vec3.h"
#include "mire_body_data.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using curvenet::Vec3;
namespace cm  = curvenet::cut_mesh;
namespace cml = curvenet::cut_mesh_laplacian;
namespace sp  = curvenet::sparse;

namespace {

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

struct DiagStats {
    double min, max;
    double abs_min_nonzero, abs_max;
    std::size_t n_zero, n_negative;
};

DiagStats summarize(const std::vector<double> &diag) {
    DiagStats s;
    s.min = 1e300; s.max = -1e300;
    s.abs_min_nonzero = 1e300; s.abs_max = 0.0;
    s.n_zero = 0; s.n_negative = 0;
    for (double d : diag) {
        if (d < s.min) s.min = d;
        if (d > s.max) s.max = d;
        if (d == 0.0) ++s.n_zero;
        else {
            const double a = std::fabs(d);
            if (a < s.abs_min_nonzero) s.abs_min_nonzero = a;
            if (a > s.abs_max)         s.abs_max = a;
            if (d < 0.0) ++s.n_negative;
        }
    }
    return s;
}

void print_diag(const char *label, const DiagStats &s, std::size_t n) {
    std::printf("  %s\n", label);
    std::printf("    n=%zu  min=%-12.4g  max=%-12.4g\n", n, s.min, s.max);
    std::printf("    abs_min_nonzero=%-12.4g  abs_max=%-12.4g  dynamic_range=%.2e\n",
                  s.abs_min_nonzero, s.abs_max,
                  (s.abs_min_nonzero > 0 ? s.abs_max / s.abs_min_nonzero : 0.0));
    std::printf("    zero_count=%zu  negative_count=%zu\n",
                  s.n_zero, s.n_negative);
}

} // namespace

int main(int argc, char **argv) {
    (void)argc; (void)argv;

    // Load Mire body from baked header.
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
    std::printf("Mire body: %zu verts, %d tris\n", nv, mire_body::n_tris);

    // Halfedge + cut mesh with 4 promoted samples.
    const curvenet::HalfedgeMesh hm =
        curvenet::halfedge_builder::from_triangles(nv, tris);
    cm::CutMesh c;
    c.base = hm;
    c.vertex_kind.assign(nv, cm::CutVertexKind::mesh_vertex_kind());
    c.segment_of_halfedge.assign(hm.he_count(), -1);
    const std::vector<Vec3> targets = {
        {  0.4,  0.0, 1.0 },
        { -0.4,  0.0, 1.0 },
        {  0.0,  0.0, 1.6 },
        {  0.0,  0.0, 0.6 },
    };
    for (std::size_t s = 0; s < targets.size(); ++s) {
        const int vi = closest_vertex(positions, targets[s]);
        c.vertex_kind[static_cast<std::size_t>(vi)] =
            cm::CutVertexKind::sample_kind(static_cast<int>(s), 0, false);
        std::printf("  sample %zu -> vertex %d  (%.3f, %.3f, %.3f)\n",
                      s, vi,
                      positions[vi].x, positions[vi].y, positions[vi].z);
    }

    std::printf("\n");

    // Assemble Lₕ (per-halfedge Laplacian) via robust path and
    // inspect its diagonal. The non-robust path produces ±∞ on this
    // mesh; the robust path uses Sharp & Crane 2020 mollification.
    const double mol_delta = cml::default_mollify_delta(positions, tris);
    std::printf("intrinsic mollification delta = %.3e\n", mol_delta);
    const sp::SparseMatrixCSR Lh   = cml::assemble_lh_csr_robust(c, positions, mol_delta);
    const sp::SparseMatrixCSR LhsM = cml::assemble_vt_lh_v_csr_robust(c, positions, mol_delta);
    std::printf("Lh   : rows=%zu  cols=%zu  nnz=%zu  mean_nnz/row=%.2f\n",
                  Lh.rows, Lh.cols, Lh.values.size(),
                  static_cast<double>(Lh.values.size()) / Lh.rows);
    std::printf("LhsM : rows=%zu  cols=%zu  nnz=%zu  mean_nnz/row=%.2f\n",
                  LhsM.rows, LhsM.cols, LhsM.values.size(),
                  static_cast<double>(LhsM.values.size()) / LhsM.rows);
    std::printf("\n");

    const std::vector<double> Lh_diag   = sp::diagonal(Lh);
    const std::vector<double> LhsM_diag = sp::diagonal(LhsM);
    print_diag("Lh diagonal:",   summarize(Lh_diag),   Lh_diag.size());
    print_diag("LhsM diagonal:", summarize(LhsM_diag), LhsM_diag.size());

    // Count cot weight magnitudes to see if the matrix has wildly
    // varying scales.
    {
        DiagStats off;
        off.min = 1e300; off.max = -1e300;
        off.abs_min_nonzero = 1e300; off.abs_max = 0.0;
        off.n_zero = 0; off.n_negative = 0;
        for (std::size_t i = 0; i < LhsM.rows; ++i) {
            const int rs = LhsM.row_ptr[i];
            const int re = LhsM.row_ptr[i + 1];
            for (int p = rs; p < re; ++p) {
                if (static_cast<std::size_t>(LhsM.col_idx[p]) == i) continue;
                const double v = LhsM.values[p];
                if (v < off.min) off.min = v;
                if (v > off.max) off.max = v;
                if (v == 0.0) ++off.n_zero;
                else {
                    const double a = std::fabs(v);
                    if (a < off.abs_min_nonzero) off.abs_min_nonzero = a;
                    if (a > off.abs_max)         off.abs_max = a;
                    if (v < 0.0) ++off.n_negative;
                }
            }
        }
        print_diag("LhsM off-diagonal entries:", off,
                     LhsM.values.size() - LhsM.rows);
    }

    return 0;
}
