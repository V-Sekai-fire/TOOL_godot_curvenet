// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
//
// Phase-3c: Schwarz as preconditioner inside outer CG. Each outer
// CG iter does one spmv against the global LhsM and one
// "schwarz_apply" (a forward + backward sweep of Schwarz over the
// meshlets, applied to the current residual). The hope: O(1) outer
// CG iters independent of nv, vs the ~200 iters multiplicative
// Schwarz alone needed.
//
// Forward + backward sweep keeps the preconditioner symmetric,
// which CG requires.
//
// Run with `make -C tests diag_meshlet_pcg`.

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

// One sweep of multiplicative Schwarz applied to the residual r.
// Returns z ≈ A^{-1} r (one full sweep).
struct MeshletDecomp {
    std::vector<std::vector<int>>          core_per_meshlet;
    std::vector<std::vector<int>>          ext_per_meshlet;
    std::vector<std::vector<std::uint8_t>> in_core_per_meshlet;
};

std::vector<double> schwarz_sweep(
        const sp::SparseMatrixCSR &A,
        const std::vector<double> &r,
        const MeshletDecomp &md,
        bool reverse) {
    const std::size_t nv = r.size();
    const std::size_t K = md.core_per_meshlet.size();
    std::vector<double> z(nv, 0.0);

    auto run_one = [&](std::size_t mi) {
        const auto &core = md.core_per_meshlet[mi];
        const auto &ext  = md.ext_per_meshlet[mi];
        const auto &in_core = md.in_core_per_meshlet[mi];
        const std::size_t n = ext.size();

        std::unordered_map<int, int> g2l;
        for (std::size_t i = 0; i < n; ++i) g2l[ext[i]] = static_cast<int>(i);

        std::vector<std::pair<std::pair<int,int>, double>> raw;
        std::vector<double> rhs(n, 0.0);
        std::vector<std::uint8_t> is_pinned(n, 0);
        std::vector<double> pin_value(n, 0.0);

        for (std::size_t i = 0; i < n; ++i) {
            const int v = ext[i];
            is_pinned[i] = in_core[v] ? 0 : 1;
            if (is_pinned[i]) {
                pin_value[i] = z[v];   // current accumulated z
            }
            rhs[i] = r[v];
        }

        for (std::size_t i = 0; i < n; ++i) {
            const int v = ext[i];
            const int rs = A.row_ptr[v];
            const int re = A.row_ptr[v + 1];
            bool has_diag = false;
            for (int k = rs; k < re; ++k) {
                const int j = A.col_idx[k];
                const auto it = g2l.find(j);
                if (it == g2l.end()) continue;
                raw.push_back({ { static_cast<int>(i), it->second }, A.values[k] });
                if (it->second == static_cast<int>(i) && A.values[k] != 0.0) has_diag = true;
            }
            if (!has_diag) {
                raw.push_back({ { static_cast<int>(i), static_cast<int>(i) }, 1.0 });
            }
        }

        std::vector<std::pair<std::pair<int,int>, double>> kept;
        for (const auto &e : raw) {
            const int i = e.first.first;
            const int j = e.first.second;
            const double val = e.second;
            if (is_pinned[i]) continue;
            if (is_pinned[j]) {
                rhs[i] -= val * pin_value[j];
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

        const std::vector<double> x_local =
            sp::cg(L, rhs, n * 4, 1e-12);
        for (std::size_t i = 0; i < n; ++i) {
            const int v = ext[i];
            if (in_core[v]) z[v] = x_local[i];
        }
        (void)core;
    };

    if (!reverse) {
        for (std::size_t m = 0; m < K; ++m) run_one(m);
    } else {
        for (std::size_t m = K; m-- > 0; ) run_one(m);
    }
    return z;
}

// Symmetric multiplicative Schwarz: forward sweep then backward.
// Output z ≈ A^{-1} r and the operator z = M^{-1} r is symmetric.
std::vector<double> schwarz_apply_symmetric(
        const sp::SparseMatrixCSR &A,
        const std::vector<double> &r,
        const MeshletDecomp &md) {
    // The forward sweep starts from z = 0 and produces z₁ ≈ A^{-1} r.
    // The backward sweep refines z₁ — but we have to feed it the
    // residual against z₁, not r. Equivalent to:
    //   z₂ = z₁ + sweep_back(r - A z₁)
    std::vector<double> z1 = schwarz_sweep(A, r, md, false);
    const std::vector<double> Az1 = sp::spmv(A, z1);
    std::vector<double> r1(r.size());
    for (std::size_t i = 0; i < r.size(); ++i) r1[i] = r[i] - Az1[i];
    std::vector<double> dz = schwarz_sweep(A, r1, md, true);
    std::vector<double> z(r.size());
    for (std::size_t i = 0; i < r.size(); ++i) z[i] = z1[i] + dz[i];
    return z;
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
    const sp::SparseMatrixCSR LhsM =
        cml::assemble_vt_lh_v_csr_robust(c, positions, mol_delta);

    std::printf("Mire body: %zu verts, %zu tris\n", nv, mire_body::n_tris);

    std::mt19937_64 rng(0xc0ffeeULL);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> y_seed(nv);
    for (auto &v : y_seed) v = dist(rng);
    const std::vector<double> b = sp::spmv(LhsM, y_seed);

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

    MeshletDecomp md;
    md.core_per_meshlet.resize(meshlets.size());
    md.ext_per_meshlet.resize(meshlets.size());
    md.in_core_per_meshlet.assign(meshlets.size(),
        std::vector<std::uint8_t>(nv, 0));

    std::vector<int> owner(nv, -1);
    for (std::size_t mi = 0; mi < meshlets.size(); ++mi) {
        const auto &ml = meshlets[mi];
        for (std::size_t j = 0; j < ml.vertex_count; ++j) {
            const int v = static_cast<int>(meshlet_vertices[ml.vertex_offset + j]);
            if (owner[v] == -1) {
                owner[v] = static_cast<int>(mi);
                md.core_per_meshlet[mi].push_back(v);
                md.in_core_per_meshlet[mi][v] = 1;
            }
        }
    }
    const auto adj = vertex_adjacency(nv, tris);
    for (std::size_t mi = 0; mi < meshlets.size(); ++mi) {
        std::unordered_set<int> ext_set;
        for (int v : md.core_per_meshlet[mi]) {
            ext_set.insert(v);
            for (int w : adj[v]) ext_set.insert(w);
        }
        md.ext_per_meshlet[mi].assign(ext_set.begin(), ext_set.end());
        std::sort(md.ext_per_meshlet[mi].begin(), md.ext_per_meshlet[mi].end());
    }
    std::printf("%zu meshlets at max_v=%zu\n\n", meshlets.size(), max_v);

    // ---- Outer Schwarz-preconditioned CG ----
    // Mirrors sp::cg but with `apply_jacobi` replaced by
    // `schwarz_apply_symmetric`.
    std::vector<double> x(nv, 0.0);
    std::vector<double> r(b);
    std::vector<double> z = schwarz_apply_symmetric(LhsM, r, md);
    std::vector<double> p = z;
    double rz_old = sp::dot(r, z);

    std::printf("Outer CG with symmetric Schwarz preconditioner:\n");
    std::printf("%-4s  %-12s  %-12s\n", "iter", "max_resid", "rz");
    std::printf("------------------------------------\n");

    const std::size_t outer_max = 30;
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
            std::printf("converged after %zu outer iters\n", outer + 1);
            break;
        }
        z = schwarz_apply_symmetric(LhsM, r, md);
        const double rz_new = sp::dot(r, z);
        const double beta = (rz_old == 0.0) ? 0.0 : (rz_new / rz_old);
        p = sp::saxpby(1.0, z, beta, p);
        rz_old = rz_new;
    }

    return 0;
}
