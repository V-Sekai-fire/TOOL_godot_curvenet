// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
//
// Two-level Schwarz preconditioner training run on Mire 81k.
//
// Per the user's two-stage instruction: validate the algorithm on
// a small training problem before committing time to the full
// 81k run. Time-capped at 30 seconds total — if a configuration
// hasn't converged by then we move on.
//
// What two-level adds over plain meshlet-Schwarz:
//   1. Forward fine sweep:    z_f^(1) = M_S^(-1) · r       (Schwarz)
//   2. Coarse correction:     y_c     = A_c^(-1) · R · (r - A z_f^(1))
//                              z_f^(2) = z_f^(1) + Rᵀ · y_c
//   3. Backward fine sweep:   z_f     = z_f^(2) + M_S^(-1) · (r - A z_f^(2))
//
// The coarse correction handles the global low-frequency modes
// 1-level Schwarz can't reach. Coarse problem is 23×23 at 5k —
// solved by direct sparse CG (small enough).

#include "curvenet/cut_mesh.h"
#include "curvenet/cut_mesh_laplacian.h"
#include "curvenet/halfedge_builder.h"
#include "curvenet/sparse_linalg.h"
#include "curvenet/two_level_schwarz.h"
#include "curvenet/vec3.h"
#include "mire_body_70k_data.h"

#include "meshoptimizer.h"

#include <algorithm>
#include <chrono>
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
namespace tls = curvenet::two_level_schwarz;

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

// One multiplicative-Schwarz sweep, returns delta = x_after - x_in.
struct SweepCtx {
    const sp::SparseMatrixCSR *G;
    const std::vector<double> *b_global;
    const std::vector<std::vector<int>> *core_per_meshlet;
    const std::vector<std::vector<int>> *ext_per_meshlet;
    const std::vector<std::vector<std::uint8_t>> *in_core_per_meshlet;
};

std::vector<double> schwarz_sweep_delta(
        const SweepCtx &ctx, const std::vector<double> &x_in) {
    std::vector<double> x = x_in;
    const std::size_t K = ctx.core_per_meshlet->size();
    for (std::size_t mi = 0; mi < K; ++mi) {
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
    }
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
    // 81k: per-iter ~16x the 5k cost (368 vs 23 meshlets). 90s cap
    // gives us time to see whether two-level convergence trajectory
    // actually contracts at this scale.
    const double T_CAP_MS = 90000.0;

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

    std::printf("Mire 81k: nv=%zu nnz=%zu\n", nv, LhsM.values.size());

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
    std::printf("%zu meshlets at max_v=%zu\n", meshlets.size(), max_v);

    SweepCtx ctx{ &LhsM, &b,
                    &core_per_meshlet, &ext_per_meshlet, &in_core_per_meshlet };

    // Build coarse map: each fine vert → its owning meshlet (= coarse node).
    std::vector<int> coarse_of(nv);
    for (std::size_t i = 0; i < nv; ++i) coarse_of[i] = owner[i];
    const std::size_t nc = meshlets.size();

    // Build Galerkin coarse matrix once.
    const double t_galerkin = now_ms();
    const sp::SparseMatrixCSR Ac = tls::galerkin_csr(coarse_of, nc, LhsM);
    std::printf("Galerkin coarse matrix: %zu x %zu, nnz=%zu, %.2f ms\n",
                  Ac.rows, Ac.cols, Ac.values.size(), now_ms() - t_galerkin);

    // ---- Two-level Schwarz outer iteration ----
    // Per outer iter:
    //   1. forward Schwarz sweep:   x ← x + delta_S(x)
    //   2. fine residual:           r = b - A x
    //   3. coarse-restrict residual: r_c = R r
    //   4. coarse solve:            y_c = A_c^(-1) r_c   (sparse CG)
    //   5. prolong + apply:         x ← x + Rᵀ y_c
    //   6. backward Schwarz sweep:  x ← x + delta_S(x)
    std::vector<double> x(nv, 0.0);
    const double t_solve_start = now_ms();
    std::printf("\nTwo-level Schwarz (max 30s):\n");
    std::printf("%-4s  %-12s  %-12s\n", "iter", "max_resid", "elapsed_ms");
    bool converged = false;
    std::size_t iters_done = 0;
    for (std::size_t outer = 0; outer < 1000; ++outer) {
        // 1. forward
        const std::vector<double> d1 = schwarz_sweep_delta(ctx, x);
        for (std::size_t i = 0; i < nv; ++i) x[i] += d1[i];

        // 2. fine residual
        const std::vector<double> Ax = sp::spmv(LhsM, x);
        std::vector<double> r(nv);
        for (std::size_t i = 0; i < nv; ++i) r[i] = b[i] - Ax[i];

        // 3-4. coarse correction
        const std::vector<double> rc = tls::restrict_fine(coarse_of, nc, r);
        const std::vector<double> yc =
            sp::cg(Ac, rc, std::max<std::size_t>(50, nc * 4), 1e-12);
        const std::vector<double> y = tls::prolong_coarse(coarse_of, nv, yc);

        // 5. apply correction
        for (std::size_t i = 0; i < nv; ++i) x[i] += y[i];

        // 6. backward sweep
        const std::vector<double> d2 = schwarz_sweep_delta(ctx, x);
        for (std::size_t i = 0; i < nv; ++i) x[i] += d2[i];

        const double r_now = max_abs_residual(LhsM, x, b);
        const double elapsed = now_ms() - t_solve_start;
        if (outer < 5 || outer % 10 == 9 || r_now < 1e-9) {
            std::printf("%-4zu  %-12.4e  %-12.0f\n", outer, r_now, elapsed);
        }
        iters_done = outer + 1;
        if (r_now < 1e-9) {
            converged = true;
            break;
        }
        if (elapsed > T_CAP_MS) {
            std::printf("time cap %.0f ms reached at iter %zu, resid %.3e\n",
                          T_CAP_MS, outer, r_now);
            break;
        }
    }
    if (converged) {
        std::printf("\nTwo-level Schwarz: converged in %zu outer iters\n",
                      iters_done);
        std::printf("vs 1-level Chebyshev-Schwarz at 5k: 670 outer iters\n");
        if (iters_done > 0) {
            const double ratio = 670.0 / static_cast<double>(iters_done);
            std::printf("speedup factor over Chebyshev-Schwarz: %.2fx\n", ratio);
        }
    } else {
        std::printf("\nTwo-level Schwarz: did NOT converge in %zu iters / %.0f ms\n",
                      iters_done, T_CAP_MS);
    }
    return converged ? 0 : 1;
}
