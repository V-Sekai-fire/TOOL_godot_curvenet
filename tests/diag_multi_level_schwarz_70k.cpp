// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
//
// Multi-level Schwarz training run on Mire 81k. Generic in level
// count: builds a hierarchy by recursive aggregation until the
// top level fits below a configurable threshold.
//
// At 5k with target_coarsening = 8 and top_threshold = 16:
//   level 0:    nv (5485)         fine
//   level 1:    23                meshlet (meshopt)
//   level 2..L: < 16              recursive principal-axis bucketing
//
// Per the user's two-stage instruction: validate the algorithm on
// the 5k training problem before committing time to the 81k full
// run. Time-capped at 30s. Compare to two-level's 193 outer iters.

#include "curvenet/cut_mesh.h"
#include "curvenet/cut_mesh_laplacian.h"
#include "curvenet/halfedge_builder.h"
#include "curvenet/multi_level_schwarz.h"
#include "curvenet/sparse_linalg.h"
#include "curvenet/two_level_schwarz.h"
#include "curvenet/vec3.h"
#include "mire_body_70k_data.h"

#include "meshoptimizer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using curvenet::Vec3;
namespace cm  = curvenet::cut_mesh;
namespace cml = curvenet::cut_mesh_laplacian;
namespace mls = curvenet::multi_level_schwarz;
namespace sp  = curvenet::sparse;
namespace tls = curvenet::two_level_schwarz;

namespace {

double now_ms() {
    using namespace std::chrono;
    return duration_cast<duration<double, std::milli>>(
        steady_clock::now().time_since_epoch()).count();
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

// Generic principal-axis aggregator: clusters n points into k
// buckets along the longest spatial extent. Returns a cmap of length
// n with values in [0, k). Works in any dimension that fits Vec3.
std::vector<int> aggregate_by_longest_axis(
        const std::vector<Vec3> &pts, std::size_t k_buckets) {
    const std::size_t n = pts.size();
    if (k_buckets == 0 || n == 0) return {};
    if (k_buckets >= n) {
        std::vector<int> cmap(n);
        for (std::size_t i = 0; i < n; ++i) cmap[i] = static_cast<int>(i);
        return cmap;
    }
    Vec3 lo{ 1e300, 1e300, 1e300 }, hi{ -1e300, -1e300, -1e300 };
    for (const auto &p : pts) {
        lo.x = std::min(lo.x, p.x); lo.y = std::min(lo.y, p.y); lo.z = std::min(lo.z, p.z);
        hi.x = std::max(hi.x, p.x); hi.y = std::max(hi.y, p.y); hi.z = std::max(hi.z, p.z);
    }
    const double sx = hi.x - lo.x, sy = hi.y - lo.y, sz = hi.z - lo.z;
    const int axis = (sx >= sy && sx >= sz) ? 0 : (sy >= sz ? 1 : 2);
    std::vector<double> proj(n);
    for (std::size_t i = 0; i < n; ++i) {
        proj[i] = axis == 0 ? pts[i].x : (axis == 1 ? pts[i].y : pts[i].z);
    }
    std::vector<std::size_t> idx(n);
    for (std::size_t i = 0; i < n; ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(),
                 [&](std::size_t a, std::size_t b) { return proj[a] < proj[b]; });
    std::vector<int> cmap(n);
    for (std::size_t r = 0; r < n; ++r) {
        const std::size_t bucket = (r * k_buckets) / n;
        cmap[idx[r]] = static_cast<int>(bucket);
    }
    return cmap;
}

struct MultiLevel {
    std::vector<std::vector<int>>     cmaps;     // length L
    std::vector<sp::SparseMatrixCSR> mats;       // length L+1
    std::vector<std::size_t>           sizes;     // length L+1
};

// Given level-i positions, aggregate to level-(i+1) by applying a
// cmap (sum-and-divide centroid). Used to produce level-i+2
// aggregations from level-i+1 positions.
std::vector<Vec3> centroids_after_cmap(const std::vector<Vec3> &pts,
                                          const std::vector<int> &cmap,
                                          std::size_t k_coarse) {
    std::vector<Vec3>   sum(k_coarse, { 0.0, 0.0, 0.0 });
    std::vector<double> cnt(k_coarse, 0.0);
    for (std::size_t i = 0; i < pts.size(); ++i) {
        const int c = cmap[i];
        sum[c].x += pts[i].x; sum[c].y += pts[i].y; sum[c].z += pts[i].z;
        cnt[c] += 1.0;
    }
    for (std::size_t i = 0; i < k_coarse; ++i) {
        const double s = cnt[i] > 0.0 ? 1.0 / cnt[i] : 0.0;
        sum[i].x *= s; sum[i].y *= s; sum[i].z *= s;
    }
    return sum;
}

void jacobi_smooth(const sp::SparseMatrixCSR &A, const std::vector<double> &b,
                   std::vector<double> &x, std::size_t n_sweeps,
                   double omega = 0.7) {
    const std::size_t n = b.size();
    std::vector<double> diag(n, 1.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (int k = A.row_ptr[i]; k < A.row_ptr[i + 1]; ++k) {
            if (A.col_idx[k] == static_cast<int>(i)) {
                diag[i] = (A.values[k] != 0.0) ? A.values[k] : 1.0;
            }
        }
    }
    for (std::size_t s = 0; s < n_sweeps; ++s) {
        const std::vector<double> Ax = sp::spmv(A, x);
        for (std::size_t i = 0; i < n; ++i) {
            x[i] += omega * (b[i] - Ax[i]) / diag[i];
        }
    }
}

// Generic V-cycle: smooth/restrict on the way down, direct-solve the
// coarsest, prolong/smooth on the way up. Works for any number of
// levels >= 1 (1 level == direct CG on the fine grid; >= 2 levels
// recurse via the cmaps).
std::vector<double> v_cycle(const MultiLevel &ml,
                            const std::vector<double> &r0,
                            std::size_t pre_smooth = 3,
                            std::size_t post_smooth = 3) {
    const std::size_t L = ml.cmaps.size();
    std::vector<std::vector<double>> rhs(L + 1);
    std::vector<std::vector<double>> sol(L + 1);
    rhs[0] = r0;
    for (std::size_t i = 0; i + 1 < L + 1; ++i) {
        sol[i].assign(ml.sizes[i], 0.0);
        if (i > 0) jacobi_smooth(ml.mats[i], rhs[i], sol[i], pre_smooth);
        const std::vector<double> Ay = sp::spmv(ml.mats[i], sol[i]);
        std::vector<double> r(ml.sizes[i]);
        for (std::size_t k = 0; k < ml.sizes[i]; ++k) r[k] = rhs[i][k] - Ay[k];
        rhs[i + 1] = tls::restrict_fine(ml.cmaps[i], ml.sizes[i + 1], r);
    }
    sol[L] = sp::cg(ml.mats[L], rhs[L],
                       std::max<std::size_t>(64, ml.sizes[L] * 8), 1e-12);
    for (std::size_t j = 0; j < L; ++j) {
        const std::size_t i = L - 1 - j;
        const std::vector<double> y_up =
            tls::prolong_coarse(ml.cmaps[i], ml.sizes[i], sol[i + 1]);
        for (std::size_t k = 0; k < ml.sizes[i]; ++k) sol[i][k] += y_up[k];
        if (i > 0) jacobi_smooth(ml.mats[i], rhs[i], sol[i], post_smooth);
    }
    return sol[0];
}

} // namespace

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    const double T_CAP_MS = 90000.0;
    // Generic hierarchy parameters: keep coarsening until the top
    // level has < TOP_THRESHOLD nodes. COARSEN_RATIO controls how
    // aggressively we shrink each level above level 1.
    const std::size_t TOP_THRESHOLD = 16;
    const std::size_t COARSEN_RATIO = 4;

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
    const sp::SparseMatrixCSR LhsM =
        cml::assemble_vt_lh_v_csr_robust(c, positions, mol_delta);

    std::printf("Mire 81k: nv=%zu nnz=%zu\n", nv, LhsM.values.size());

    std::mt19937_64 rng(0xc0ffeeULL);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> y_seed(nv);
    for (auto &v : y_seed) v = dist(rng);
    const std::vector<double> b = sp::spmv(LhsM, y_seed);

    // Level 0 -> 1: meshlet aggregation (the connectivity-aware level).
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
    const std::size_t k_meshlets = meshopt_buildMeshlets(
        meshlets.data(), meshlet_vertices.data(), meshlet_triangles.data(),
        indices.data(), indices.size(), mp.data(), nv, sizeof(float) * 3,
        max_v, max_t, 0.0f);
    meshlets.resize(k_meshlets);

    std::vector<int> owner(nv, -1);
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
    const std::size_t n1 = meshlets.size();

    SweepCtx ctx{ &LhsM, &b,
                    &core_per_meshlet, &ext_per_meshlet, &in_core_per_meshlet };

    std::vector<int> cmap0(nv);
    for (std::size_t i = 0; i < nv; ++i) cmap0[i] = owner[i];

    // Build the multilevel hierarchy generically: keep aggregating
    // by principal-axis bucketing until size <= TOP_THRESHOLD.
    MultiLevel ml;
    ml.cmaps.push_back(cmap0);
    ml.sizes.push_back(nv);
    ml.sizes.push_back(n1);

    std::vector<Vec3> centroids = centroids_after_cmap(positions, cmap0, n1);
    while (ml.sizes.back() > TOP_THRESHOLD) {
        const std::size_t n_cur = ml.sizes.back();
        const std::size_t n_next = std::max<std::size_t>(2, n_cur / COARSEN_RATIO);
        const std::vector<int> cmap_up =
            aggregate_by_longest_axis(centroids, n_next);
        ml.cmaps.push_back(cmap_up);
        ml.sizes.push_back(n_next);
        centroids = centroids_after_cmap(centroids, cmap_up, n_next);
    }
    std::printf("hierarchy: ");
    for (std::size_t s : ml.sizes) std::printf("%zu ", s);
    std::printf("(%zu levels)\n", ml.sizes.size());

    // Build Galerkin matrices bottom-up.
    const double t_g = now_ms();
    ml.mats.push_back(LhsM);
    for (std::size_t lv = 0; lv < ml.cmaps.size(); ++lv) {
        const sp::SparseMatrixCSR &A = ml.mats.back();
        ml.mats.push_back(tls::galerkin_csr(ml.cmaps[lv], ml.sizes[lv + 1], A));
    }
    std::printf("Galerkin chain: ");
    for (const auto &A : ml.mats) std::printf("%zu(nnz=%zu) ", A.rows, A.values.size());
    std::printf("[%.2f ms]\n", now_ms() - t_g);

    // ---- Multi-level Schwarz outer iteration ----
    std::vector<double> x(nv, 0.0);
    const double t_solve_start = now_ms();
    std::printf("\n%zu-level Schwarz V-cycle (max %.0fs):\n",
                  ml.sizes.size(), T_CAP_MS / 1000.0);
    std::printf("%-4s  %-12s  %-12s\n", "iter", "max_resid", "elapsed_ms");
    bool converged = false;
    std::size_t iters_done = 0;
    for (std::size_t outer = 0; outer < 1000; ++outer) {
        const std::vector<double> d1 = schwarz_sweep_delta(ctx, x);
        for (std::size_t i = 0; i < nv; ++i) x[i] += d1[i];

        const std::vector<double> Ax = sp::spmv(LhsM, x);
        std::vector<double> r(nv);
        for (std::size_t i = 0; i < nv; ++i) r[i] = b[i] - Ax[i];

        const std::vector<double> y = v_cycle(ml, r);
        for (std::size_t i = 0; i < nv; ++i) x[i] += y[i];

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
            std::printf("time cap reached at iter %zu, resid %.3e\n",
                          outer, r_now);
            break;
        }
    }
    if (converged) {
        std::printf("\n%zu-level Schwarz: converged in %zu outer iters\n",
                      ml.sizes.size(), iters_done);
        std::printf("vs 2-level: 193, vs Chebyshev-Schwarz: 137\n");
    } else {
        std::printf("\n%zu-level Schwarz: did NOT converge in %zu iters / %.0f ms\n",
                      ml.sizes.size(), iters_done, T_CAP_MS);
    }
    return converged ? 0 : 1;
}
