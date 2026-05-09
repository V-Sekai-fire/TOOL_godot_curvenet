// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
//
// Phase-3b: additive Schwarz with 1-ring overlap. Each meshlet's
// "core" C_m is its original vertex set (~256 verts). The local
// system is built over the *extended* set V_m_ext = C_m ∪ 1-ring(C_m)
// — verts in V_m_ext but outside C_m are pinned to the current
// iterate. After every meshlet solves, the global x is updated
// only at C_m for the meshlet that "owns" each vert.
//
// Each vert has a unique owning meshlet (the first that contains
// it), so aggregation is unambiguous: no averaging.
//
// Why this works where Phase 3a didn't: the verts at meshlet
// boundaries are now DOFs in their owning meshlet (not pinned),
// while NEIGHBOURING meshlets see them as pinned. Information
// flows across boundaries via the overlap.
//
// Run with `make -C tests diag_meshlet_schwarz_overlap`.

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

// Compute vertex adjacency (1-ring) from triangle indices. For each
// vertex v, adj[v] is the sorted set of verts that share a triangle
// edge with v.
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

// Local system over an extended vertex set, with symmetric Dirichlet
// at verts outside the core.
struct LocalSystem {
    sp::SparseMatrixCSR        A;
    std::vector<double>        rhs;
    std::vector<unsigned int>  global_id;       // local i → global vert
    std::vector<std::uint8_t>  is_pinned;       // true ↔ outside core
    std::vector<std::uint8_t>  is_core;         // true ↔ inside V_m core
};

LocalSystem build_local_overlap(
        const sp::SparseMatrixCSR &G,
        const std::vector<double> &b_global,
        const std::vector<double> &pin_source,
        const std::vector<int> &core_verts,
        const std::vector<int> &ext_verts,
        const std::vector<std::uint8_t> &in_core) {
    LocalSystem ls;
    const std::size_t n = ext_verts.size();
    ls.global_id.resize(n);
    ls.is_pinned.assign(n, 0);
    ls.is_core.assign(n, 0);
    ls.rhs.assign(n, 0.0);

    std::unordered_map<int, int> g2l;
    for (std::size_t i = 0; i < n; ++i) {
        ls.global_id[i] = static_cast<unsigned int>(ext_verts[i]);
        g2l[ext_verts[i]] = static_cast<int>(i);
        ls.is_core[i] = in_core[ext_verts[i]];
        ls.is_pinned[i] = !ls.is_core[i] ? 1 : 0;
        ls.rhs[i] = b_global[ext_verts[i]];
    }
    std::vector<double> pin_value(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        if (ls.is_pinned[i]) pin_value[i] = pin_source[ls.global_id[i]];
    }
    (void)core_verts;

    // Slice G by ext_verts.
    std::vector<std::pair<std::pair<int, int>, double>> raw;
    for (std::size_t i = 0; i < n; ++i) {
        const int v = ext_verts[i];
        const int rs = G.row_ptr[v];
        const int re = G.row_ptr[v + 1];
        bool has_diag = false;
        for (int k = rs; k < re; ++k) {
            const int j = G.col_idx[k];
            const auto it = g2l.find(j);
            if (it == g2l.end()) continue;
            raw.push_back({ { static_cast<int>(i), it->second }, G.values[k] });
            if (it->second == static_cast<int>(i) && G.values[k] != 0.0) has_diag = true;
        }
        if (!has_diag) {
            raw.push_back({ { static_cast<int>(i), static_cast<int>(i) }, 1.0 });
        }
    }

    // Symmetric Dirichlet at pinned (non-core) verts.
    std::vector<std::pair<std::pair<int, int>, double>> kept;
    for (const auto &e : raw) {
        const int i = e.first.first;
        const int j = e.first.second;
        const double v = e.second;
        if (ls.is_pinned[i]) continue;
        if (ls.is_pinned[j]) {
            ls.rhs[i] -= v * pin_value[j];
            continue;
        }
        kept.push_back(e);
    }
    for (std::size_t i = 0; i < n; ++i) {
        if (ls.is_pinned[i]) {
            kept.push_back({ { static_cast<int>(i), static_cast<int>(i) }, 1.0 });
            ls.rhs[i] = pin_value[i];
        }
    }
    std::sort(kept.begin(), kept.end());
    std::vector<std::pair<std::pair<int, int>, double>> merged;
    for (const auto &p : kept) {
        if (!merged.empty() && merged.back().first == p.first) {
            merged.back().second += p.second;
        } else {
            merged.push_back(p);
        }
    }
    ls.A.rows = n; ls.A.cols = n;
    ls.A.row_ptr.assign(n + 1, 0);
    for (const auto &p : merged) ++ls.A.row_ptr[p.first.first + 1];
    for (std::size_t i = 0; i < n; ++i) ls.A.row_ptr[i + 1] += ls.A.row_ptr[i];
    ls.A.col_idx.resize(merged.size());
    ls.A.values.resize(merged.size());
    std::vector<int> cursor = ls.A.row_ptr;
    for (const auto &p : merged) {
        const int idx = cursor[p.first.first]++;
        ls.A.col_idx[idx] = p.first.second;
        ls.A.values[idx]  = p.second;
    }
    return ls;
}

} // namespace

int main(int argc, char **argv) {
    (void)argc; (void)argv;

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
    sp::SparseMatrixCSR LhsM =
        cml::assemble_vt_lh_v_csr_robust(c, positions, mol_delta);

    // Regularize sample slots.
    {
        std::vector<std::pair<std::pair<int,int>, double>> coo;
        std::vector<std::uint8_t> is_sample_row(nv, 0);
        for (std::size_t i = 0; i < nv; ++i) {
            const int rs = LhsM.row_ptr[i];
            const int re = LhsM.row_ptr[i + 1];
            bool any_nonzero = false;
            for (int k = rs; k < re; ++k) {
                if (LhsM.values[k] != 0.0) { any_nonzero = true; break; }
            }
            if (!any_nonzero) is_sample_row[i] = 1;
        }
        for (std::size_t i = 0; i < nv; ++i) {
            if (is_sample_row[i]) {
                coo.push_back({ { static_cast<int>(i), static_cast<int>(i) }, 1.0 });
                continue;
            }
            const int rs = LhsM.row_ptr[i];
            const int re = LhsM.row_ptr[i + 1];
            for (int k = rs; k < re; ++k) {
                coo.push_back({ { static_cast<int>(i), LhsM.col_idx[k] },
                                 LhsM.values[k] });
            }
        }
        LhsM.row_ptr.assign(nv + 1, 0);
        for (auto &p : coo) ++LhsM.row_ptr[p.first.first + 1];
        for (std::size_t i = 0; i < nv; ++i) LhsM.row_ptr[i + 1] += LhsM.row_ptr[i];
        LhsM.col_idx.assign(coo.size(), 0);
        LhsM.values.assign(coo.size(), 0.0);
        std::vector<int> cursor = LhsM.row_ptr;
        for (auto &p : coo) {
            const int idx = cursor[p.first.first]++;
            LhsM.col_idx[idx] = p.first.second;
            LhsM.values[idx]  = p.second;
        }
    }

    std::printf("Mire body: %zu verts, %zu tris\n", nv, mire_body::n_tris);

    std::mt19937_64 rng(0xc0ffeeULL);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> y_seed(nv);
    for (auto &v : y_seed) v = dist(rng);
    const std::vector<double> b = sp::spmv(LhsM, y_seed);
    const std::vector<double> x_global =
        sp::cg(LhsM, b, std::max<std::size_t>(2000, nv), 1e-10);

    // Meshlet partition.
    std::vector<float> mp(nv * 3);
    for (std::size_t i = 0; i < nv * 3; ++i) mp[i] = mire_body::positions[i];
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

    // ---- Determine each meshlet's "core" verts and the global owner.
    // owner[v] = the meshlet that solves for v as a DOF (not pinned).
    // We pick the first meshlet that contains v in iteration order.
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
    // Sanity: every vert has an owner.
    std::size_t total_core = 0;
    for (const auto &c : core_per_meshlet) total_core += c.size();
    std::printf("%zu meshlets, sum of core sizes = %zu (= %zu = nv)\n\n",
                  meshlets.size(), total_core, nv);

    // ---- 1-ring extension per meshlet.
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
    // Track core membership per global vert.
    std::vector<std::vector<std::uint8_t>> in_core_per_meshlet(meshlets.size(),
        std::vector<std::uint8_t>(nv, 0));
    for (std::size_t mi = 0; mi < meshlets.size(); ++mi) {
        for (int v : core_per_meshlet[mi]) in_core_per_meshlet[mi][v] = 1;
    }

    // Stats: how big are the extended sets?
    std::size_t total_ext = 0;
    std::size_t max_ext = 0, max_core = 0;
    for (std::size_t mi = 0; mi < meshlets.size(); ++mi) {
        const std::size_t e = ext_per_meshlet[mi].size();
        const std::size_t cc = core_per_meshlet[mi].size();
        total_ext += e;
        if (e > max_ext) max_ext = e;
        if (cc > max_core) max_core = cc;
    }
    std::printf("core sizes:     mean=%.1f  max=%zu\n",
                  static_cast<double>(total_core) / meshlets.size(), max_core);
    std::printf("extended sizes: mean=%.1f  max=%zu  (overhead=%.2fx)\n\n",
                  static_cast<double>(total_ext) / meshlets.size(),
                  max_ext, static_cast<double>(total_ext) / total_core);

    // ---- Multiplicative Schwarz with 1-ring overlap ----
    // Each meshlet pins its boundary to the LATEST x (not a snapshot
    // from the start of this outer iter), so subsequent meshlets in
    // the same iter see the updates. Sequential per outer iter, but
    // converges much faster than additive; for parallel-friendly
    // dispatching one would damp additive with ω≈0.5 instead.
    std::vector<double> x(nv, 0.0);

    std::printf("Multiplicative Schwarz with 1-ring overlap:\n");
    std::printf("%-4s  %-12s  %-12s  %-12s\n",
                  "iter", "max_err", "rms_err", "delta_x");
    std::printf("------------------------------------------------------\n");

    const std::size_t max_outer = 80;
    for (std::size_t outer = 0; outer < max_outer; ++outer) {
        std::vector<double> x_at_start(x);

        for (std::size_t mi = 0; mi < meshlets.size(); ++mi) {
            LocalSystem ls = build_local_overlap(
                LhsM, b, x,                 // <- live x, not snapshot
                core_per_meshlet[mi],
                ext_per_meshlet[mi],
                in_core_per_meshlet[mi]);
            const std::size_t local_max_iter = ls.A.rows * 4;
            const std::vector<double> x_local =
                sp::cg(ls.A, ls.rhs, local_max_iter, 1e-12);
            for (std::size_t i = 0; i < ls.A.rows; ++i) {
                if (ls.is_core[i]) {
                    x[ls.global_id[i]] = x_local[i];
                }
            }
        }

        double max_err = 0.0;
        double sum_sq  = 0.0;
        double max_dx  = 0.0;
        for (std::size_t i = 0; i < nv; ++i) {
            const double e = std::fabs(x[i] - x_global[i]);
            const double dx = std::fabs(x[i] - x_at_start[i]);
            sum_sq += e * e;
            if (e > max_err) max_err = e;
            if (dx > max_dx) max_dx = dx;
        }
        const double rms = std::sqrt(sum_sq / nv);
        std::printf("%-4zu  %-12.4e  %-12.4e  %-12.4e\n",
                      outer, max_err, rms, max_dx);
        if (max_err < 1e-7) {
            std::printf("converged (max_err < 1e-7) after %zu outer iters\n", outer + 1);
            break;
        }
    }

    return 0;
}
