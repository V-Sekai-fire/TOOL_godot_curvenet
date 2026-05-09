// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
//
// Phase-2 meshlet diagnostic: solve each meshlet independently with
// boundary verts pinned to the global solution's values, and check
// that the per-meshlet interior matches the global solution within
// fp32 tolerance. This is the "best case" for the per-meshlet
// decomposition — what if we had a perfect oracle for the boundary?
// If the per-meshlet solves track the global one here, then any
// later approach that approximates the boundary (Schwarz, Schur)
// only has to drive the boundary error down.
//
// Constructs a simple synthetic problem: solve LhsM · x = b for one
// random RHS column. Comparison is in fp64 to avoid noise from the
// fp32 CG path.
//
// Run via `make -C tests diag_meshlet_solve`.

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

// Build a local LhsM by slicing the global rows+cols by `meshlet_verts`,
// then applying symmetric Dirichlet boundary conditions at `pinned_verts`
// (in our case, verts that appear in >1 meshlet).
//
// Symmetric pinning: zero out BOTH row i and column i of pinned i, set
// diag[i,i] = 1, set rhs[i] = pin_value[i], and for each non-pinned j
// where A[j, i] != 0 in the original sliced matrix, subtract
// A[j, i] * pin_value[i] from rhs[j]. This preserves SPD so plain CG
// works (vs asymmetric row-only replacement which breaks SPD and CG
// diverges).
//
// Caller passes pin_values[] in *local* order (length = n_verts).
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
                          const std::vector<double> &x_global_for_pinning) {
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
            pin_value[i] = x_global_for_pinning[meshlet_verts[i]];
        }
        ls.rhs[i] = b_global[meshlet_verts[i]];
    }

    // Step 1: collect sliced (i, j, value) ignoring pinning.
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
        // Sample slots in the global matrix had zero rows; the global
        // pre-regularization replaced them with identity. Same here:
        // ensure diag exists.
        if (!has_diag) {
            raw.push_back({ { static_cast<int>(i), static_cast<int>(i) }, 1.0 });
        }
    }

    // Step 2: apply symmetric Dirichlet at pinned verts.
    //   * for each non-pinned j and pinned i: rhs[j] -= A[j, i] * pin_value[i],
    //                                         then A[j, i] = 0
    //   * for each pinned i: zero row i, set A[i, i] = 1, rhs[i] = pin_value[i]
    std::vector<std::pair<std::pair<int, int>, double>> kept;
    for (const auto &e : raw) {
        const int i = e.first.first;
        const int j = e.first.second;
        const double v = e.second;
        if (ls.is_pinned[i]) {
            // Will be replaced with identity row below; drop original entries.
            continue;
        }
        if (ls.is_pinned[j]) {
            // RHS adjustment: b[i] -= A[i,j] * pin[j]
            ls.rhs[i] -= v * pin_value[j];
            // Drop A[i,j] (column eliminated).
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

    // ---- Setup: same as the other Mire diags ----
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

    // Regularize: replace zero-row (sample slot) rows with identity so
    // the system is invertible. This is what the deformer's runtime
    // does implicitly via the C constraint — for this synthetic test
    // it lets CG converge.
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

    // ---- Synthetic RHS in the range of A ----
    // Use b = A · y for some random y. Then b is guaranteed in range
    // and CG converges.
    std::mt19937_64 rng(0xc0ffeeULL);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> y_seed(nv);
    for (auto &v : y_seed) v = dist(rng);
    const std::vector<double> b = sp::spmv(LhsM, y_seed);

    // ---- Global solve (fp64 reference) ----
    // The global LhsM has a 4-dim kernel (the 4 sample slots have
    // zero rows + zero RHS, by construction). CG against such a
    // singular system still converges if the RHS is in the range
    // of A; our zero-out above guarantees it. Use a high iter cap.
    const std::size_t global_max_iter = std::max<std::size_t>(2000, nv);
    const std::vector<double> x_global =
        sp::cg(LhsM, b, global_max_iter, 1e-10);
    // Residual of the global solve.
    {
        const std::vector<double> Ax = sp::spmv(LhsM, x_global);
        double max_r = 0.0;
        for (std::size_t i = 0; i < nv; ++i) {
            const double r = std::fabs(Ax[i] - b[i]);
            if (r > max_r) max_r = r;
        }
        std::printf("Global solve: max_residual=%.3e\n", max_r);
    }

    // ---- Meshlet partition ----
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

    // ---- Per-meshlet solves with boundary pinned to global ----
    // Aggregate per-meshlet solutions into a global vector by:
    //   * boundary verts: average across meshlets that contain them
    //   * interior verts: take the unique meshlet's solution
    std::vector<double> x_meshlet(nv, 0.0);
    std::vector<int>    coverage(nv, 0);

    double sum_iters = 0.0;
    std::size_t worst_iters = 0;
    double max_interior_err = 0.0;
    int    max_interior_idx = -1;

    for (std::size_t mi = 0; mi < meshlets.size(); ++mi) {
        const auto &ml = meshlets[mi];
        std::vector<unsigned int> verts(
            meshlet_vertices.begin() + ml.vertex_offset,
            meshlet_vertices.begin() + ml.vertex_offset + ml.vertex_count);
        LocalSystem ls = build_local(LhsM, b, verts, seen_count, x_global);

        const std::size_t local_max_iter = ls.A.rows * 4;
        const std::vector<double> x_local =
            sp::cg(ls.A, ls.rhs, local_max_iter, 1e-10);
        // No iter-count return from sp::cg, but we can detect
        // convergence by residual.
        const std::vector<double> Ax = sp::spmv(ls.A, x_local);
        double max_r = 0.0;
        for (std::size_t i = 0; i < ls.A.rows; ++i) {
            const double r = std::fabs(Ax[i] - ls.rhs[i]);
            if (r > max_r) max_r = r;
        }
        sum_iters += static_cast<double>(local_max_iter); // bound, not actual

        // Aggregate into global vector.
        for (std::size_t i = 0; i < ls.A.rows; ++i) {
            const unsigned int g = ls.global_id[i];
            x_meshlet[g] += x_local[i];
            coverage[g]  += 1;

            if (!ls.is_pinned[i]) {
                const double err = std::fabs(x_local[i] - x_global[g]);
                if (err > max_interior_err) {
                    max_interior_err = err;
                    max_interior_idx = static_cast<int>(g);
                }
            }
        }
        (void)max_r; (void)worst_iters;
    }
    // Average over coverage to handle boundary verts seen in multiple
    // meshlets.
    for (std::size_t i = 0; i < nv; ++i) {
        if (coverage[i] > 0) x_meshlet[i] /= coverage[i];
    }

    // Per-vert error meshlet vs global.
    double max_err = 0.0;
    int max_err_idx = -1;
    double sum_sq = 0.0;
    for (std::size_t i = 0; i < nv; ++i) {
        const double e = std::fabs(x_meshlet[i] - x_global[i]);
        sum_sq += e * e;
        if (e > max_err) { max_err = e; max_err_idx = static_cast<int>(i); }
    }
    const double rms = std::sqrt(sum_sq / nv);

    std::printf("Per-meshlet solve (boundary pinned to global):\n");
    std::printf("  max interior error (vs global):  %.3e at vert %d\n",
                  max_interior_err, max_interior_idx);
    std::printf("  max combined error (incl bnd):   %.3e at vert %d\n",
                  max_err, max_err_idx);
    std::printf("  rms error:                        %.3e\n", rms);
    std::printf("\n");
    std::printf("If the interior error is at fp64 floor (~1e-9), the per-meshlet\n");
    std::printf("decomposition is consistent and Phase 3 only has to provide\n");
    std::printf("a good boundary approximation (Schwarz / Schur).\n");
    return 0;
}
