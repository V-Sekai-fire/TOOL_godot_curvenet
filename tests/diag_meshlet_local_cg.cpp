// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
//
// Per Gall's law cycle: validate the claim "per-meshlet matrices
// have κ ≤ 10² and fp32 CG converges fine on them." Built the
// architecture argument for per-meshlet GPU CG on this assumption;
// before designing on top, measure whether fp64 CG on per-meshlet
// locals actually has small iter counts.
//
// What this measures:
//   * for each meshlet at 81k, builds the local sliced+Dirichlet
//     matrix (same as diag_meshlet_pcg_chol_70k's build_local_cache)
//   * runs sparse::cg_diag against a synthetic SPD-projected RHS
//     at tol=1e-8
//   * reports per-meshlet iter count, residual
//   * aggregates: total iters across all meshlets, max iters,
//     mean iters
//
// If max per-meshlet iters << total monolithic iters (20k), the
// per-meshlet hypothesis holds and the GPU per-meshlet path is
// architecturally sound. If they're comparable, per-meshlet doesn't
// help convergence on its own and we need stronger preconditioning
// inside each local solve.

#include "curvenet/cut_mesh.h"
#include "curvenet/cut_mesh_laplacian.h"
#include "curvenet/halfedge_builder.h"
#include "curvenet/sparse_linalg.h"
#include "curvenet/vec3.h"
#include "mire_body_70k_data.h"

#include "meshoptimizer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <random>
#include <unordered_map>
#include <unordered_set>
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

std::vector<std::vector<int>>
vertex_adjacency(std::size_t n_verts, const std::vector<int> &tris) {
    std::vector<std::unordered_set<int>> sets(n_verts);
    for (std::size_t t = 0; t + 2 < tris.size(); t += 3) {
        const int a = tris[t + 0];
        const int b = tris[t + 1];
        const int c = tris[t + 2];
        sets[a].insert(b); sets[a].insert(c);
        sets[b].insert(a); sets[b].insert(c);
        sets[c].insert(a); sets[c].insert(b);
    }
    std::vector<std::vector<int>> adj(n_verts);
    for (std::size_t v = 0; v < n_verts; ++v) {
        adj[v].assign(sets[v].begin(), sets[v].end());
        std::sort(adj[v].begin(), adj[v].end());
    }
    return adj;
}

// Build local sliced + symmetric-Dirichlet-pinned matrix for one
// meshlet. Returns CSR + an arbitrary RHS that's in the matrix's
// range, so per-meshlet CG has a defined converged answer.
struct LocalSystem {
    sp::SparseMatrixCSR A;
    std::vector<double> b;
};

LocalSystem build_local(const sp::SparseMatrixCSR &G,
                          const std::vector<int> &ext,
                          const std::vector<std::uint8_t> &in_core) {
    const std::size_t n = ext.size();
    std::unordered_map<int, int> g2l;
    std::vector<std::uint8_t> is_pinned(n, 0);
    for (std::size_t i = 0; i < n; ++i) {
        g2l[ext[i]] = static_cast<int>(i);
        is_pinned[i] = in_core[ext[i]] ? 0 : 1;
    }

    std::vector<double> Adense(n * n, 0.0);
    std::vector<std::uint8_t> has_diag(n, 0);
    for (std::size_t i = 0; i < n; ++i) {
        const int v = ext[i];
        const int rs = G.row_ptr[v];
        const int re = G.row_ptr[v + 1];
        for (int kk = rs; kk < re; ++kk) {
            const int j_global = G.col_idx[kk];
            const auto it = g2l.find(j_global);
            if (it == g2l.end()) continue;
            const int j_local = it->second;
            if (is_pinned[i]) continue;
            if (is_pinned[j_local]) continue;
            Adense[i * n + j_local] += G.values[kk];
            if (j_local == static_cast<int>(i) && G.values[kk] != 0.0) has_diag[i] = 1;
        }
    }
    for (std::size_t i = 0; i < n; ++i) {
        if (is_pinned[i] || !has_diag[i]) Adense[i * n + i] = 1.0;
    }
    // Tikhonov regularization (same as the meshlet pcg path)
    const double tikhonov = 1e-8;
    for (std::size_t i = 0; i < n; ++i) Adense[i * n + i] += tikhonov;

    // Convert dense → CSR (skipping zeros).
    LocalSystem ls;
    std::vector<cml::CooEntry> coo;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            const double v = Adense[i * n + j];
            if (v != 0.0) {
                coo.push_back({ static_cast<int>(i), static_cast<int>(j), v });
            }
        }
    }
    ls.A = cml::coo_to_csr(std::move(coo), n, n);

    // RHS: project a random vector through the matrix to keep it in range.
    std::mt19937_64 rng(0xc0ffee01ULL);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> y(n);
    for (auto &v : y) v = dist(rng);
    ls.b = sp::spmv(ls.A, y);
    return ls;
}

} // namespace

int main(int argc, char **argv) {
    (void)argc; (void)argv;

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
    const std::vector<Vec3> sample_targets = {
        {  0.4,  0.0, 1.0 }, { -0.4,  0.0, 1.0 },
        {  0.0,  0.0, 1.6 }, {  0.0,  0.0, 0.6 },
    };
    for (std::size_t s = 0; s < sample_targets.size(); ++s) {
        const int vi = closest_vertex(positions, sample_targets[s]);
        c.vertex_kind[static_cast<std::size_t>(vi)] =
            cm::CutVertexKind::sample_kind(static_cast<int>(s), 0, false);
    }
    const double mol_delta = cml::default_mollify_delta(positions, tris);
    const sp::SparseMatrixCSR LhsM =
        cml::assemble_vt_lh_v_csr_robust(c, positions, mol_delta);
    std::printf("Mire 81k: nv=%zu nh=%zu LhsM nnz=%zu\n",
                  nv, hm.he_count(), LhsM.values.size());

    // Meshlet partition.
    std::vector<float> mp(nv * 3);
    for (std::size_t i = 0; i < nv * 3; ++i) mp[i] = mire_body_70k::positions[i];
    std::vector<unsigned int> indices(mire_body_70k::n_tris * 3);
    for (std::size_t i = 0; i < mire_body_70k::n_tris * 3; ++i) {
        indices[i] = static_cast<unsigned int>(mire_body_70k::tris[i]);
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

    std::vector<int>   owner(nv, -1);
    std::vector<std::vector<int>> core_per_meshlet(meshlets.size());
    for (std::size_t mi = 0; mi < meshlets.size(); ++mi) {
        const auto &ml = meshlets[mi];
        for (std::size_t j = 0; j < ml.vertex_count; ++j) {
            const int v = static_cast<int>(meshlet_vertices[ml.vertex_offset + j]);
            if (owner[v] == -1) {
                owner[v] = static_cast<int>(mi);
                core_per_meshlet[mi].push_back(v);
            }
        }
    }
    std::vector<std::vector<std::uint8_t>> in_core_per_meshlet(meshlets.size(),
        std::vector<std::uint8_t>(nv, 0));
    for (std::size_t mi = 0; mi < meshlets.size(); ++mi) {
        for (int v : core_per_meshlet[mi]) in_core_per_meshlet[mi][v] = 1;
    }
    const auto adj = vertex_adjacency(nv, tris);
    std::vector<std::vector<int>> ext_per_meshlet(meshlets.size());
    for (std::size_t mi = 0; mi < meshlets.size(); ++mi) {
        std::unordered_set<int> ext_set;
        for (int v : core_per_meshlet[mi]) {
            ext_set.insert(v);
            for (int w : adj[v]) ext_set.insert(w);
        }
        ext_per_meshlet[mi].assign(ext_set.begin(), ext_set.end());
        std::sort(ext_per_meshlet[mi].begin(), ext_per_meshlet[mi].end());
    }
    std::printf("%zu meshlets at max_v=%zu\n\n", meshlets.size(), max_v);

    // ---- Per-meshlet CG iter-count distribution ----
    std::vector<std::size_t> iters_dist;
    std::size_t total_iters = 0;
    std::size_t max_iters = 0;
    std::size_t min_iters = SIZE_MAX;
    std::size_t failed_meshlets = 0;
    for (std::size_t mi = 0; mi < meshlets.size(); ++mi) {
        LocalSystem ls = build_local(LhsM, ext_per_meshlet[mi],
                                       in_core_per_meshlet[mi]);
        sp::CgStats stats;
        const std::size_t cap = ls.A.rows * 5;
        const std::vector<double> x =
            sp::cg_diag(ls.A, ls.b, cap, 1e-8, stats);
        if (std::sqrt(stats.final_rr) >= 1e-8) ++failed_meshlets;
        iters_dist.push_back(stats.iters);
        total_iters += stats.iters;
        if (stats.iters > max_iters) max_iters = stats.iters;
        if (stats.iters < min_iters) min_iters = stats.iters;
        (void)x;
    }
    const double mean_iters = static_cast<double>(total_iters) / meshlets.size();

    std::sort(iters_dist.begin(), iters_dist.end());
    const std::size_t p50 = iters_dist[iters_dist.size() / 2];
    const std::size_t p90 = iters_dist[(iters_dist.size() * 9) / 10];

    std::printf("Per-meshlet fp64 CG iter counts (tol=1e-8):\n");
    std::printf("  meshlets:          %zu\n", meshlets.size());
    std::printf("  failed to converge: %zu (residual >= 1e-8 at iter cap)\n",
                  failed_meshlets);
    std::printf("  iters min:         %zu\n", min_iters);
    std::printf("  iters max:         %zu\n", max_iters);
    std::printf("  iters mean:        %.1f\n", mean_iters);
    std::printf("  iters p50:         %zu\n", p50);
    std::printf("  iters p90:         %zu\n", p90);
    std::printf("  total iters:       %zu (across all meshlets)\n",
                  total_iters);
    std::printf("\n");
    std::printf("Comparison: monolithic CG at 81k = 20,668 iters\n");
    std::printf("Per-meshlet path is %.1fx more iters total than monolithic\n",
                  static_cast<double>(total_iters) / 20668.0);
    return 0;
}
