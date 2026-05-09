// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
//
// Phase-1 meshlet diagnostic. Partitions the Mire body into meshlets
// of 256 verts each, builds the global `LhsM_csr` (the matrix CG
// actually solves against), and slices it per-meshlet to inspect
// what the per-meshlet sub-matrices look like.
//
// The constant-work hypothesis: each per-meshlet sub-matrix is small
// (~256x256), sparse (mean nnz/row ≈ 7 from triangle adjacency), and
// well-conditioned (similar dynamic range to the global matrix). If
// that holds, per-meshlet CG should converge in O(√κ_local) iters,
// which is bounded as the global mesh grows.
//
// Reports per-meshlet:
//   * size, nnz, mean nnz/row
//   * diagonal min/max/dynamic range
//   * sample-promoted vert count (for solver constraint forcing)
//   * boundary-vert count (verts shared with other meshlets)
//
// Run via `make -C tests diag_meshlet_lhsm`.

#include "curvenet/cut_mesh.h"
#include "curvenet/cut_mesh_laplacian.h"
#include "curvenet/halfedge_builder.h"
#include "curvenet/sparse_linalg.h"
#include "curvenet/vec3.h"
#include "mire_body_data.h"

#include "meshoptimizer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
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

struct PerMeshlet {
    std::size_t n_verts;
    std::size_t n_interior;        // verts unique to this meshlet
    std::size_t n_boundary;        // verts shared with other meshlets
    std::size_t n_samples;         // sample-promoted verts in this meshlet
    std::size_t nnz;
    double      mean_nnz_per_row;
    double      diag_min, diag_max;
    double      diag_abs_min_nonzero;
    double      diag_dynamic_range;
    std::size_t diag_zero;
    std::size_t diag_negative;
};

PerMeshlet inspect_meshlet(const sp::SparseMatrixCSR &LhsM,
                              const std::vector<unsigned int> &meshlet_verts,
                              const std::vector<std::uint8_t> &seen_count,
                              const std::vector<std::uint8_t> &is_sample) {
    PerMeshlet p{};
    p.n_verts = meshlet_verts.size();

    // Build a fast "is in this meshlet" lookup.
    std::vector<std::uint8_t> in_meshlet(LhsM.rows, 0);
    for (unsigned int v : meshlet_verts) in_meshlet[v] = 1;

    p.diag_min = 1e300; p.diag_max = -1e300;
    p.diag_abs_min_nonzero = 1e300;

    for (unsigned int v : meshlet_verts) {
        if (seen_count[v] > 1) ++p.n_boundary;
        else                   ++p.n_interior;
        if (is_sample[v]) ++p.n_samples;

        // Walk row v, count nnzs that stay in the meshlet.
        const int rs = LhsM.row_ptr[v];
        const int re = LhsM.row_ptr[v + 1];
        for (int k = rs; k < re; ++k) {
            const int j = LhsM.col_idx[k];
            if (j < 0 || static_cast<std::size_t>(j) >= LhsM.rows) continue;
            if (!in_meshlet[j]) continue;
            const double val = LhsM.values[k];
            ++p.nnz;
            if (static_cast<unsigned int>(j) == v) {
                if (val < p.diag_min) p.diag_min = val;
                if (val > p.diag_max) p.diag_max = val;
                if (val == 0.0) ++p.diag_zero;
                else {
                    const double a = std::fabs(val);
                    if (a < p.diag_abs_min_nonzero) p.diag_abs_min_nonzero = a;
                    if (val < 0.0) ++p.diag_negative;
                }
            }
        }
    }
    p.mean_nnz_per_row = static_cast<double>(p.nnz) / p.n_verts;
    if (p.diag_abs_min_nonzero < 1e300 && p.diag_max != 0.0) {
        const double abs_max = std::max(std::fabs(p.diag_min), std::fabs(p.diag_max));
        p.diag_dynamic_range = abs_max / p.diag_abs_min_nonzero;
    }
    return p;
}

} // namespace

int main(int argc, char **argv) {
    (void)argc; (void)argv;

    // Load Mire body.
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

    // ---- Cut-mesh + global LhsM (matches the deformer's bind path) ----
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
    std::vector<std::uint8_t> is_sample(nv, 0);
    for (std::size_t s = 0; s < sample_targets.size(); ++s) {
        const int vi = closest_vertex(positions, sample_targets[s]);
        c.vertex_kind[static_cast<std::size_t>(vi)] =
            cm::CutVertexKind::sample_kind(static_cast<int>(s), 0, false);
        is_sample[static_cast<std::size_t>(vi)] = 1;
    }
    const double mol_delta = cml::default_mollify_delta(positions, tris);
    const sp::SparseMatrixCSR LhsM =
        cml::assemble_vt_lh_v_csr_robust(c, positions, mol_delta);

    std::printf("Mire body: %zu verts, %zu tris\n", nv, mire_body::n_tris);
    std::printf("LhsM (global): rows=%zu  nnz=%zu  mean_nnz/row=%.2f\n\n",
                  LhsM.rows, LhsM.values.size(),
                  static_cast<double>(LhsM.values.size()) / LhsM.rows);

    // ---- Meshlet partition ----
    std::vector<float> mp(nv * 3);
    for (std::size_t i = 0; i < nv * 3; ++i) {
        mp[i] = mire_body::positions[i];
    }
    std::vector<unsigned int> indices(mire_body::n_tris * 3);
    for (std::size_t i = 0; i < mire_body::n_tris * 3; ++i) {
        indices[i] = static_cast<unsigned int>(mire_body::tris[i]);
    }
    const std::size_t max_v = 256, max_t = 512;
    const std::size_t bound = meshopt_buildMeshletsBound(indices.size(), max_v, max_t);
    std::vector<meshopt_Meshlet> meshlets(bound);
    std::vector<unsigned int>    meshlet_vertices (bound * max_v);
    std::vector<unsigned char>   meshlet_triangles(bound * max_t * 3);
    const std::size_t k = meshopt_buildMeshlets(
        meshlets.data(), meshlet_vertices.data(), meshlet_triangles.data(),
        indices.data(), indices.size(), mp.data(), nv, sizeof(float) * 3,
        max_v, max_t, 0.0f);
    meshlets.resize(k);

    // Boundary count = verts in >1 meshlet.
    std::vector<std::uint8_t> seen_count(nv, 0);
    for (const auto &ml : meshlets) {
        for (std::size_t j = 0; j < ml.vertex_count; ++j) {
            const unsigned int v = meshlet_vertices[ml.vertex_offset + j];
            if (seen_count[v] < 255) ++seen_count[v];
        }
    }

    // ---- Per-meshlet inspection ----
    std::printf("Per-meshlet sub-matrices (rows+cols sliced from global LhsM):\n");
    std::printf("%-3s %-5s %-5s %-5s %-3s %-6s %-6s %-12s %-12s %-12s\n",
                  "#", "n_v", "intr", "bnd", "smp",
                  "nnz", "nnz/r", "diag_min", "diag_max", "dyn_range");
    std::printf("--------------------------------------------------------------------------------\n");

    std::size_t total_nnz_local = 0;
    std::size_t meshlets_with_samples = 0;
    std::size_t meshlets_zero_diag = 0;
    double max_dyn_range = 0.0;
    for (std::size_t i = 0; i < meshlets.size(); ++i) {
        const auto &ml = meshlets[i];
        std::vector<unsigned int> verts(
            meshlet_vertices.begin() + ml.vertex_offset,
            meshlet_vertices.begin() + ml.vertex_offset + ml.vertex_count);
        const PerMeshlet p = inspect_meshlet(LhsM, verts, seen_count, is_sample);
        total_nnz_local += p.nnz;
        if (p.n_samples > 0) ++meshlets_with_samples;
        if (p.diag_zero > 0)  ++meshlets_zero_diag;
        if (p.diag_dynamic_range > max_dyn_range) max_dyn_range = p.diag_dynamic_range;
        std::printf("%-3zu %-5zu %-5zu %-5zu %-3zu %-6zu %-6.2f %-12.4g %-12.4g %-12.2e\n",
                      i, p.n_verts, p.n_interior, p.n_boundary, p.n_samples,
                      p.nnz, p.mean_nnz_per_row,
                      p.diag_min, p.diag_max, p.diag_dynamic_range);
    }

    std::printf("\nSummary:\n");
    std::printf("  total local nnz:              %zu  (vs %zu in the global matrix)\n",
                  total_nnz_local, LhsM.values.size());
    std::printf("  meshlets containing a sample: %zu / %zu\n",
                  meshlets_with_samples, meshlets.size());
    std::printf("  meshlets with a 0 diagonal:   %zu\n", meshlets_zero_diag);
    std::printf("  worst per-meshlet dyn range:  %.2e\n", max_dyn_range);
    return 0;
}
