// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
//
// Phase-3 diagnostic: Schwarz iteration drives the per-meshlet
// decomposition toward the global solution *without* needing the
// global x as oracle. Each outer iter solves every meshlet
// independently with boundary verts pinned to the current iterate
// (additive Schwarz), then combines: average at shared boundary
// verts, take-the-only-value at interior verts.
//
// Reports convergence trajectory: max error vs global solve per
// outer iter. The slope tells us whether Schwarz is fast enough
// (per-frame budget allows ~10-20 outer iters) or whether we need
// 1-ring overlap / multiplicative variant / Schur complement.
//
// Run via `make -C tests diag_meshlet_schwarz`.

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

// Reused from Phase 2: build local sliced matrix with symmetric
// Dirichlet pinning at boundary verts (those in >1 meshlet).
struct LocalSystem {
    sp::SparseMatrixCSR        A;
    std::vector<double>        rhs;
    std::vector<unsigned int>  global_id;
    std::unordered_map<unsigned int, int> g2l;
    std::vector<std::uint8_t>  is_pinned;
};

LocalSystem build_local(const sp::SparseMatrixCSR &G,
                          const std::vector<double> &b_global,
                          const std::vector<unsigned int> &meshlet_verts,
                          const std::vector<std::uint8_t> &seen_count,
                          const std::vector<double> &pin_source) {
    LocalSystem ls;
    const std::size_t n = meshlet_verts.size();
    ls.global_id = meshlet_verts;
    ls.is_pinned.assign(n, 0);
    ls.rhs.assign(n, 0.0);
    std::vector<double> pin_value(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        ls.g2l[meshlet_verts[i]] = static_cast<int>(i);
        if (seen_count[meshlet_verts[i]] > 1) {
            ls.is_pinned[i] = 1;
            pin_value[i] = pin_source[meshlet_verts[i]];
        }
        ls.rhs[i] = b_global[meshlet_verts[i]];
    }

    std::vector<std::pair<std::pair<int, int>, double>> raw;
    for (std::size_t i = 0; i < n; ++i) {
        const unsigned int v = meshlet_verts[i];
        const int rs = G.row_ptr[v];
        const int re = G.row_ptr[v + 1];
        bool has_diag = false;
        for (int k = rs; k < re; ++k) {
            const int j_global = G.col_idx[k];
            const auto it = ls.g2l.find(static_cast<unsigned int>(j_global));
            if (it == ls.g2l.end()) continue;
            const int j_local = it->second;
            raw.push_back({ { static_cast<int>(i), j_local }, G.values[k] });
            if (j_local == static_cast<int>(i) && G.values[k] != 0.0) has_diag = true;
        }
        if (!has_diag) {
            raw.push_back({ { static_cast<int>(i), static_cast<int>(i) }, 1.0 });
        }
    }
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

    // Regularize sample slots to identity rows.
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

    // Synthetic SPD-projected RHS so global CG converges cleanly.
    std::mt19937_64 rng(0xc0ffeeULL);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> y_seed(nv);
    for (auto &v : y_seed) v = dist(rng);
    const std::vector<double> b = sp::spmv(LhsM, y_seed);

    // Reference monolithic solve.
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

    std::vector<std::uint8_t> seen_count(nv, 0);
    for (const auto &ml : meshlets) {
        for (std::size_t j = 0; j < ml.vertex_count; ++j) {
            const unsigned int v = meshlet_vertices[ml.vertex_offset + j];
            if (seen_count[v] < 255) ++seen_count[v];
        }
    }
    std::printf("%zu meshlets at max_v=%zu\n\n", meshlets.size(), max_v);

    // ---- Schwarz outer iteration ----
    // x starts at zero; each outer iter solves every meshlet locally
    // with boundary pinned to current x, then averages overlapping
    // values back into x.
    std::vector<double> x(nv, 0.0);
    std::vector<double> x_next(nv, 0.0);
    std::vector<int>    coverage(nv, 0);

    std::printf("Schwarz outer iteration (synthetic random RHS, in range of A):\n");
    std::printf("%-4s  %-12s  %-12s  %-12s\n",
                  "iter", "max_err", "rms_err", "delta_x");
    std::printf("------------------------------------------------------\n");

    const std::size_t max_outer = 30;
    for (std::size_t outer = 0; outer < max_outer; ++outer) {
        std::fill(x_next.begin(), x_next.end(), 0.0);
        std::fill(coverage.begin(), coverage.end(), 0);

        for (std::size_t mi = 0; mi < meshlets.size(); ++mi) {
            const auto &ml = meshlets[mi];
            std::vector<unsigned int> verts(
                meshlet_vertices.begin() + ml.vertex_offset,
                meshlet_vertices.begin() + ml.vertex_offset + ml.vertex_count);
            LocalSystem ls = build_local(LhsM, b, verts, seen_count, x);
            const std::size_t local_max_iter = ls.A.rows * 4;
            const std::vector<double> x_local =
                sp::cg(ls.A, ls.rhs, local_max_iter, 1e-12);
            for (std::size_t i = 0; i < ls.A.rows; ++i) {
                const unsigned int g = ls.global_id[i];
                x_next[g]   += x_local[i];
                coverage[g] += 1;
            }
        }

        // Average over coverage (boundary verts get arithmetic mean).
        double max_err = 0.0;
        double sum_sq  = 0.0;
        double max_dx  = 0.0;
        for (std::size_t i = 0; i < nv; ++i) {
            if (coverage[i] > 0) x_next[i] /= coverage[i];
            const double e = std::fabs(x_next[i] - x_global[i]);
            const double dx = std::fabs(x_next[i] - x[i]);
            sum_sq += e * e;
            if (e > max_err) max_err = e;
            if (dx > max_dx) max_dx = dx;
        }
        const double rms = std::sqrt(sum_sq / nv);
        std::printf("%-4zu  %-12.4e  %-12.4e  %-12.4e\n",
                      outer, max_err, rms, max_dx);
        x.swap(x_next);
        if (max_dx < 1e-10) {
            std::printf("converged (delta < 1e-10)\n");
            break;
        }
    }

    return 0;
}
