// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
//
// Wraps the per-meshlet multiplicative-Schwarz outer iteration with
// Wang 2015's Chebyshev acceleration (`curvenet::chebyshev_accel`).
// Phase 3b without the Chebyshev wrapper (diag_meshlet_schwarz_overlap)
// converged in residual but at ~0.97/iter, needing ~670 outer iters
// to reach 1e-9 on the synthetic SPD-projected RHS. Chebyshev with
// the right ρ estimate should drop that to ~150-200.
//
// Mire body 5485 verts, since the smaller mesh measures faster.

#include "curvenet/chebyshev_accel.h"
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
namespace cb  = curvenet::chebyshev_accel;
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

// One multiplicative Schwarz sweep over all meshlets, starting from
// x_in (snapshotted) and returning the swept-through global x. Used
// as the base step inside Chebyshev; we report Δ = x_out - x_in.
struct SweepCtx {
    const sp::SparseMatrixCSR *G;
    const std::vector<double> *b_global;
    const std::vector<std::vector<int>> *core_per_meshlet;
    const std::vector<std::vector<int>> *ext_per_meshlet;
    const std::vector<std::vector<std::uint8_t>> *in_core_per_meshlet;
};

std::vector<double> schwarz_sweep_delta(
        const SweepCtx &ctx, const std::vector<double> &x_in) {
    std::vector<double> x = x_in;   // local mutable copy
    const std::size_t K = ctx.core_per_meshlet->size();
    for (std::size_t mi = 0; mi < K; ++mi) {
        const auto &core = (*ctx.core_per_meshlet)[mi];
        const auto &ext  = (*ctx.ext_per_meshlet)[mi];
        const auto &in_core = (*ctx.in_core_per_meshlet)[mi];
        const std::size_t n = ext.size();

        std::unordered_map<int, int> g2l;
        for (std::size_t i = 0; i < n; ++i) g2l[ext[i]] = static_cast<int>(i);

        std::vector<std::pair<std::pair<int,int>, double>> raw;
        std::vector<double> rhs(n, 0.0);
        std::vector<std::uint8_t> is_pinned(n, 0);
        std::vector<double> pin_value(n, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            is_pinned[i] = in_core[ext[i]] ? 0 : 1;
            if (is_pinned[i]) pin_value[i] = x[ext[i]];
            rhs[i] = (*ctx.b_global)[ext[i]];
        }
        for (std::size_t i = 0; i < n; ++i) {
            const int v = ext[i];
            const int rs = ctx.G->row_ptr[v];
            const int re = ctx.G->row_ptr[v + 1];
            bool has_diag = false;
            for (int k = rs; k < re; ++k) {
                const int j = ctx.G->col_idx[k];
                const auto it = g2l.find(j);
                if (it == g2l.end()) continue;
                raw.push_back({ { static_cast<int>(i), it->second }, ctx.G->values[k] });
                if (it->second == static_cast<int>(i) && ctx.G->values[k] != 0.0) has_diag = true;
            }
            if (!has_diag) raw.push_back({ { static_cast<int>(i), static_cast<int>(i) }, 1.0 });
        }
        std::vector<std::pair<std::pair<int,int>, double>> kept;
        for (const auto &e : raw) {
            const int i = e.first.first;
            const int j = e.first.second;
            const double v = e.second;
            if (is_pinned[i]) continue;
            if (is_pinned[j]) {
                rhs[i] -= v * pin_value[j];
                continue;
            }
            kept.push_back(e);
        }
        for (std::size_t i = 0; i < n; ++i) {
            if (is_pinned[i]) {
                kept.push_back({ { static_cast<int>(i), static_cast<int>(i) }, 1.0 });
                rhs[i] = pin_value[i];
            }
        }
        std::sort(kept.begin(), kept.end());
        std::vector<std::pair<std::pair<int,int>, double>> merged;
        for (const auto &p : kept) {
            if (!merged.empty() && merged.back().first == p.first) {
                merged.back().second += p.second;
            } else {
                merged.push_back(p);
            }
        }
        sp::SparseMatrixCSR L;
        L.rows = n; L.cols = n;
        L.row_ptr.assign(n + 1, 0);
        for (const auto &p : merged) ++L.row_ptr[p.first.first + 1];
        for (std::size_t i = 0; i < n; ++i) L.row_ptr[i + 1] += L.row_ptr[i];
        L.col_idx.resize(merged.size());
        L.values.resize(merged.size());
        std::vector<int> cursor = L.row_ptr;
        for (const auto &p : merged) {
            const int idx = cursor[p.first.first]++;
            L.col_idx[idx] = p.first.second;
            L.values[idx]  = p.second;
        }
        const std::vector<double> x_local = sp::cg(L, rhs, n * 4, 1e-12);
        for (std::size_t i = 0; i < n; ++i) {
            if (in_core[ext[i]]) x[ext[i]] = x_local[i];
        }
        (void)core;
    }
    // Δ = x - x_in
    std::vector<double> delta(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) delta[i] = x[i] - x_in[i];
    return delta;
}

double max_abs_residual(const sp::SparseMatrixCSR &A,
                          const std::vector<double> &x,
                          const std::vector<double> &b) {
    const std::vector<double> Ax = sp::spmv(A, x);
    double m = 0.0;
    for (std::size_t i = 0; i < b.size(); ++i) {
        m = std::max(m, std::fabs(Ax[i] - b[i]));
    }
    return m;
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

    std::printf("Mire body 81k: nv=%zu LhsM nnz=%zu\n", nv, LhsM.values.size());

    std::mt19937_64 rng(0xc0ffeeULL);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> y_seed(nv);
    for (auto &v : y_seed) v = dist(rng);
    const std::vector<double> b = sp::spmv(LhsM, y_seed);

    // Meshlet partition + 1-ring extension.
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

    SweepCtx ctx{ &LhsM, &b,
                    &core_per_meshlet, &ext_per_meshlet, &in_core_per_meshlet };
    std::printf("%zu meshlets at max_v=%zu\n\n", meshlets.size(), max_v);

    // ---- Plain (unaccelerated) outer Schwarz, for comparison ----
    std::size_t plain_iters = 0;
    bool plain_converged = false;
    {
        std::vector<double> x(nv, 0.0);
        std::printf("Plain multiplicative Schwarz (no Chebyshev):\n");
        for (std::size_t outer = 0; outer < 800; ++outer) {
            const std::vector<double> delta = schwarz_sweep_delta(ctx, x);
            for (std::size_t i = 0; i < nv; ++i) x[i] += delta[i];
            const double r = max_abs_residual(LhsM, x, b);
            if (outer % 50 == 49) std::printf("  iter %-4zu  resid %.3e\n", outer, r);
            if (r < 1e-9) {
                plain_iters = outer + 1;
                plain_converged = true;
                break;
            }
        }
        if (plain_converged)
            std::printf("plain: converged after %zu iters\n\n", plain_iters);
        else
            std::printf("plain: did not converge in 800 iters\n\n");
    }

    // ---- Chebyshev-accelerated outer Schwarz ----
    auto step = [&](const std::vector<double> &x_in) -> std::vector<double> {
        return schwarz_sweep_delta(ctx, x_in);
    };

    // Try a few rho values; the unaccelerated path showed ~0.97/iter
    // contraction at this scale, so rho ≈ 0.97 is the right spectral
    // estimate. Tune empirically.
    std::printf("Chebyshev-accelerated outer Schwarz, sweeping rho:\n");
    for (const double rho : { 0.95, 0.97, 0.98, 0.985, 0.99, 0.993, 0.995 }) {
        std::vector<double> x(nv, 0.0);
        std::vector<double> x_prev = x;
        std::vector<double> x_curr = x;
        double omega = 1.0;
        std::size_t reached = 0;
        bool converged = false;
        for (std::size_t outer = 0; outer < 800; ++outer) {
            const std::vector<double> delta = schwarz_sweep_delta(ctx, x_curr);
            std::vector<double> x_next(nv, 0.0);
            if (outer == 0) {
                for (std::size_t i = 0; i < nv; ++i) x_next[i] = x_curr[i] + delta[i];
                omega = cb::omega1(rho);
            } else {
                omega = cb::omega_next(rho, omega);
                for (std::size_t i = 0; i < nv; ++i) {
                    const double a = delta[i] + x_curr[i] - x_prev[i];
                    x_next[i] = omega * a + x_prev[i];
                }
            }
            x_prev = x_curr;
            x_curr = x_next;
            const double r = max_abs_residual(LhsM, x_curr, b);
            if (r < 1e-9) {
                reached = outer + 1;
                converged = true;
                break;
            }
        }
        if (converged) {
            const double speedup = plain_converged
                ? static_cast<double>(plain_iters) / static_cast<double>(reached)
                : 0.0;
            std::printf("  rho=%.3f  iters=%-4zu  speedup vs plain: %.2fx\n",
                          rho, reached, speedup);
        } else {
            std::printf("  rho=%.3f  did not converge in 800 iters\n", rho);
        }
    }
    return 0;
}
