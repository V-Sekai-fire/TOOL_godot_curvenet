// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
//
// Phase-3d: Schwarz-preconditioned outer CG with each meshlet's
// local matrix Cholesky-factored once at bind time. Per Schwarz
// apply, the per-meshlet step is now O(n²) back-substitution,
// not O(n²·iter) inner CG. Same outer convergence rate as
// diag_meshlet_pcg.cpp but each outer iter is ~10× cheaper, so
// the total per-frame cost lands in the 11 ms budget at our
// 70k-vert target.
//
// Structure:
//   * bind: meshlet partition; per-meshlet build local sliced
//     matrix as dense; Cholesky factor it (the diagonal-1
//     "regularization" for sample slots and pinned outer-ring
//     verts keeps the local matrix SPD).
//   * apply (M⁻¹·r → z): for each meshlet, build local RHS by
//     restricting r and adjusting for pinned values (current z
//     at outer ring), back-substitute through the cached factor,
//     write back to z[core_m].
//   * outer CG: standard PCG using this apply as preconditioner.

#include "curvenet/cut_mesh.h"
#include "curvenet/cut_mesh_laplacian.h"
#include "curvenet/dense_linalg.h"
#include "curvenet/halfedge_builder.h"
#include "curvenet/sparse_linalg.h"
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
namespace dl  = curvenet::dense;
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

// One meshlet's cached state: local index mapping, pinning mask,
// dense Cholesky factor of the (post-Dirichlet) local matrix, and
// the sparse A[i, pinned_j] entries we need for per-apply RHS
// adjustment.
struct LocalCache {
    std::vector<int>           ext;            // global vert ids in extension
    std::unordered_map<int,int> g2l;           // global → local index
    std::vector<std::uint8_t>  is_pinned;      // length = ext.size()
    std::vector<std::uint8_t>  is_core;        // length = ext.size(), 1 if in V_m
    std::vector<double>        L;              // dense Cholesky factor (n×n)
    std::size_t                n;
    // Per-apply RHS adjustment: for each (i_local, j_local) where i is
    // a DOF and j is pinned, store A[i,j] so we can do
    //   rhs[i] -= A[i,j] * z[ext[j]]
    // at apply time without rebuilding the matrix.
    struct PinDep {
        int i_local;
        int j_local;
        double a_ij;
    };
    std::vector<PinDep>        pin_deps;
};

LocalCache build_local_cache(
        const sp::SparseMatrixCSR &G,
        const std::vector<int> &core,
        const std::vector<int> &ext,
        const std::vector<std::uint8_t> &in_core) {
    LocalCache c;
    c.ext = ext;
    c.n = ext.size();
    c.is_pinned.assign(c.n, 0);
    c.is_core.assign(c.n, 0);
    for (std::size_t i = 0; i < c.n; ++i) {
        c.g2l[ext[i]] = static_cast<int>(i);
        c.is_core[i] = in_core[ext[i]] ? 1 : 0;
        c.is_pinned[i] = c.is_core[i] ? 0 : 1;
    }
    (void)core;

    // Slice G by ext, then symmetric Dirichlet at pinned slots.
    std::vector<double> Adense(c.n * c.n, 0.0);
    std::vector<std::uint8_t> has_diag(c.n, 0);
    for (std::size_t i = 0; i < c.n; ++i) {
        const int v = ext[i];
        const int rs = G.row_ptr[v];
        const int re = G.row_ptr[v + 1];
        for (int kk = rs; kk < re; ++kk) {
            const int j_global = G.col_idx[kk];
            const auto it = c.g2l.find(j_global);
            if (it == c.g2l.end()) continue;
            const int j_local = it->second;
            // Symmetric Dirichlet: drop entries in pinned rows (replaced
            // with identity below); record A[i,j] for pinned-j as a
            // pin dependency for RHS adjustment.
            if (c.is_pinned[i]) continue;
            if (c.is_pinned[j_local]) {
                c.pin_deps.push_back({ static_cast<int>(i), j_local, G.values[kk] });
                continue;
            }
            Adense[i * c.n + j_local] += G.values[kk];
            if (j_local == static_cast<int>(i) && G.values[kk] != 0.0) has_diag[i] = 1;
        }
    }
    // Identity rows for pinned slots and zero-row sample slots.
    for (std::size_t i = 0; i < c.n; ++i) {
        if (c.is_pinned[i] || !has_diag[i]) {
            Adense[i * c.n + i] = 1.0;
        }
    }
    // Tikhonov regularization: add tiny epsilon to all diagonals so
    // Cholesky doesn't fail on near-PSD blocks where the cot Laplacian's
    // condition number happens to be large after slicing.
    const double tikhonov = 1e-8;
    for (std::size_t i = 0; i < c.n; ++i) {
        Adense[i * c.n + i] += tikhonov;
    }
    // Sanity-check: count non-positive diagonals in Adense before
    // Cholesky. An SPD matrix has all positive diagonals; if we see
    // zero/negative diagonals here, Cholesky will fail at sqrt.
    static int debug_meshlet_idx = 0;
    int neg_diag = 0;
    int zero_diag = 0;
    double min_diag = 1e300;
    for (std::size_t i = 0; i < c.n; ++i) {
        const double d = Adense[i * c.n + i];
        if (d < 0.0) ++neg_diag;
        else if (d == 0.0) ++zero_diag;
        if (d < min_diag) min_diag = d;
    }
    const std::size_t pinned_count =
        static_cast<std::size_t>(std::count(c.is_pinned.begin(),
                                              c.is_pinned.end(),
                                              std::uint8_t{1}));
    std::printf("  meshlet %d: n=%zu pinned=%zu neg_diag=%d zero_diag=%d min_diag=%.3e\n",
                  debug_meshlet_idx, c.n, pinned_count,
                  neg_diag, zero_diag, min_diag);
    ++debug_meshlet_idx;
    c.L = dl::cholesky_factor(c.n, Adense);
    return c;
}

// Apply one meshlet's local solve given current z (which holds
// values at the outer-ring "pinned" verts) and global residual r.
// Updates z at this meshlet's core verts only.
void apply_local(const LocalCache &c,
                   const std::vector<double> &r,
                   std::vector<double> &z) {
    std::vector<double> rhs(c.n, 0.0);
    for (std::size_t i = 0; i < c.n; ++i) {
        if (c.is_pinned[i]) {
            // Pinned slots: identity row, RHS = current z[pinned vert].
            rhs[i] = z[c.ext[i]];
        } else {
            rhs[i] = r[c.ext[i]];
        }
    }
    // Symmetric Dirichlet RHS adjustment.
    for (const auto &pd : c.pin_deps) {
        rhs[pd.i_local] -= pd.a_ij * z[c.ext[pd.j_local]];
    }
    const std::vector<double> x_local = dl::solve_with_cholesky(c.n, c.L, rhs);
    for (std::size_t i = 0; i < c.n; ++i) {
        if (c.is_core[i]) z[c.ext[i]] = x_local[i];
    }
}

// Symmetric multiplicative Schwarz preconditioner: forward sweep,
// then backward sweep against the post-forward residual.
std::vector<double> schwarz_apply_sym(
        const sp::SparseMatrixCSR &A,
        const std::vector<double> &r,
        const std::vector<LocalCache> &caches) {
    const std::size_t nv = r.size();
    std::vector<double> z(nv, 0.0);
    for (const auto &c : caches) apply_local(c, r, z);
    // Backward sweep on the residual against z so far.
    const std::vector<double> Az = sp::spmv(A, z);
    std::vector<double> r2(nv);
    for (std::size_t i = 0; i < nv; ++i) r2[i] = r[i] - Az[i];
    std::vector<double> dz(nv, 0.0);
    for (std::size_t mi = caches.size(); mi-- > 0; ) {
        apply_local(caches[mi], r2, dz);
    }
    for (std::size_t i = 0; i < nv; ++i) z[i] += dz[i];
    return z;
}

double now_ms() {
    using namespace std::chrono;
    return duration_cast<duration<double, std::milli>>(
        steady_clock::now().time_since_epoch()).count();
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

    std::printf("Mire body: %zu verts, %zu tris\n", nv, mire_body_70k::n_tris);

    std::mt19937_64 rng(0xc0ffeeULL);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> y_seed(nv);
    for (auto &v : y_seed) v = dist(rng);
    const std::vector<double> b = sp::spmv(LhsM, y_seed);

    // Meshlet partition + cores + 1-ring extensions.
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

    // ---- Bind: build + Cholesky-factor each local once ----
    const double t_bind = now_ms();
    std::vector<LocalCache> caches(meshlets.size());
    std::size_t nan_count = 0;
    double worst_diag = 1e300;
    std::size_t worst_meshlet = 0;
    for (std::size_t mi = 0; mi < meshlets.size(); ++mi) {
        caches[mi] = build_local_cache(
            LhsM, core_per_meshlet[mi], ext_per_meshlet[mi],
            in_core_per_meshlet[mi]);
        // Check the Cholesky factor's diagonal for nan / non-positive
        // values (signal that the local matrix wasn't SPD).
        for (std::size_t i = 0; i < caches[mi].n; ++i) {
            const double d = caches[mi].L[i * caches[mi].n + i];
            if (std::isnan(d) || std::isinf(d)) ++nan_count;
            if (d < worst_diag) { worst_diag = d; worst_meshlet = mi; }
        }
    }
    const double bind_ms = now_ms() - t_bind;
    std::printf("%zu meshlets, bind (build+Cholesky) = %.2f ms\n",
                  meshlets.size(), bind_ms);
    std::printf("Cholesky diag stats: nan_count=%zu  worst_diag=%.3e (in meshlet %zu)\n\n",
                  nan_count, worst_diag, worst_meshlet);

    // ---- Outer Schwarz-preconditioned CG ----
    std::vector<double> x(nv, 0.0);
    std::vector<double> r(b);
    const double t_solve = now_ms();
    std::vector<double> z = schwarz_apply_sym(LhsM, r, caches);
    std::vector<double> p = z;
    double rz_old = sp::dot(r, z);

    std::printf("Outer CG with Cholesky-cached Schwarz preconditioner:\n");
    std::printf("%-4s  %-12s  %-12s\n", "iter", "max_resid", "rz");
    std::printf("------------------------------------\n");

    const std::size_t outer_max = 60;
    std::size_t iters_used = outer_max;
    for (std::size_t outer = 0; outer < outer_max; ++outer) {
        const std::vector<double> Ap = sp::spmv(LhsM, p);
        const double pAp = sp::dot(p, Ap);
        if (pAp == 0.0) break;
        const double alpha = rz_old / pAp;
        sp::axpy_inplace(alpha, p, x);
        sp::axpy_inplace(-alpha, Ap, r);

        double max_resid = 0.0;
        for (std::size_t i = 0; i < nv; ++i) {
            if (std::fabs(r[i]) > max_resid) max_resid = std::fabs(r[i]);
        }
        std::printf("%-4zu  %-12.4e  %-12.4e\n", outer, max_resid, rz_old);
        if (max_resid < 1e-9) {
            iters_used = outer + 1;
            break;
        }
        z = schwarz_apply_sym(LhsM, r, caches);
        const double rz_new = sp::dot(r, z);
        const double beta = (rz_old == 0.0) ? 0.0 : (rz_new / rz_old);
        p = sp::saxpby(1.0, z, beta, p);
        rz_old = rz_new;
    }
    const double solve_ms = now_ms() - t_solve;
    std::printf("\nsolve_ms total: %.2f ms over %zu outer iters (%.2f ms / iter)\n",
                  solve_ms, iters_used,
                  solve_ms / static_cast<double>(iters_used));
    return 0;
}
