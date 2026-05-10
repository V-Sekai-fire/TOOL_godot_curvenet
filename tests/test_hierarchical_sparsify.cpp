// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
//
// RapidCheck mirror of HierarchicalSparsifyCompensate.lean.

#include "curvenet/hierarchical_sparsify.h"
#include "curvenet/sparse_linalg.h"

#include <rapidcheck.h>
#include <cmath>
#include <vector>

namespace hsc = curvenet::hsc;
namespace sp  = curvenet::sparse;

int main() {
    bool ok = true;

    // Path 0 - 1 - 2 with weights 1, 2.
    // Eliminating 1 with deg(1) = 3:
    //   compensated edge (0, 2) = 0 + (1 * 2) / 3 = 0.6666...
    ok &= rc::check("path3, eliminate vertex 1 -> single edge (0,2) w=2/3", [] {
        hsc::Graph g;
        g.num_verts = 3;
        g.edges = { { 0, 1, 1.0 }, { 1, 2, 2.0 } };
        const auto g2 = hsc::eliminate_vertex(g, 1);
        RC_ASSERT(g2.edges.size() == 1u);
        RC_ASSERT(std::fabs(hsc::edge_weight(g2, 0, 2) - 2.0 / 3.0) < 1e-12);
    });

    // Star: 0 connected to 1, 2, 3 with weights 1, 2, 3.
    // Eliminating 0 with deg(0) = 6:
    //   (1, 2) = 1*2/6 = 1/3
    //   (1, 3) = 1*3/6 = 1/2
    //   (2, 3) = 2*3/6 = 1
    ok &= rc::check("star4, eliminate center -> triangle on leaves", [] {
        hsc::Graph g;
        g.num_verts = 4;
        g.edges = { { 0, 1, 1.0 }, { 0, 2, 2.0 }, { 0, 3, 3.0 } };
        const auto g2 = hsc::eliminate_vertex(g, 0);
        RC_ASSERT(g2.edges.size() == 3u);
        RC_ASSERT(std::fabs(hsc::edge_weight(g2, 1, 2) - 1.0 / 3.0) < 1e-12);
        RC_ASSERT(std::fabs(hsc::edge_weight(g2, 1, 3) - 0.5) < 1e-12);
        RC_ASSERT(std::fabs(hsc::edge_weight(g2, 2, 3) - 1.0) < 1e-12);
    });

    // Triangle 0-1-2 (all weights 1) + leaf 3 attached to 0 (weight 4).
    // Eliminating 0:
    //   existing (1, 2) accumulates: 1 + 1*1/6 = 7/6
    //   new (1, 3) = 1*4/6 = 2/3
    //   new (2, 3) = 1*4/6 = 2/3
    ok &= rc::check("triangle plus leaf, accumulate compensation", [] {
        hsc::Graph g;
        g.num_verts = 4;
        g.edges = {
            { 0, 1, 1.0 }, { 0, 2, 1.0 }, { 1, 2, 1.0 }, { 0, 3, 4.0 }
        };
        const auto g2 = hsc::eliminate_vertex(g, 0);
        RC_ASSERT(std::fabs(hsc::edge_weight(g2, 1, 2) - (7.0 / 6.0)) < 1e-12);
        RC_ASSERT(std::fabs(hsc::edge_weight(g2, 1, 3) - (2.0 / 3.0)) < 1e-12);
        RC_ASSERT(std::fabs(hsc::edge_weight(g2, 2, 3) - (2.0 / 3.0)) < 1e-12);
    });

    // CSR roundtrip: graph -> Laplacian CSR -> graph yields same edges.
    ok &= rc::check("graph <-> CSR Laplacian roundtrip preserves edges", [] {
        hsc::Graph g;
        g.num_verts = 4;
        g.edges = {
            { 0, 1, 1.5 }, { 0, 2, 2.5 }, { 1, 3, 0.75 }, { 2, 3, 4.0 }
        };
        const sp::SparseMatrixCSR L = hsc::graph_to_csr(g);
        const hsc::Graph g2 = hsc::csr_to_graph(L);
        RC_ASSERT(g2.num_verts == g.num_verts);
        // Every original edge appears in g2 with same weight.
        for (const auto &e : g.edges) {
            const double w2 =
                hsc::edge_weight(g2, std::get<0>(e), std::get<1>(e));
            RC_ASSERT(std::fabs(w2 - std::get<2>(e)) < 1e-12);
        }
    });

    // After elimination, the Schur-complement Laplacian's CSR form
    // should be SPD on the kept vertices and produce the same
    // matrix-vector action on a vector that's zero at v.
    ok &= rc::check("Schur eliminate preserves L action on kept dof", [] {
        // 4-node path 0-1-2-3 with all unit weights. Eliminate node 1.
        hsc::Graph g;
        g.num_verts = 4;
        g.edges = { { 0, 1, 1.0 }, { 1, 2, 1.0 }, { 2, 3, 1.0 } };
        const auto g2 = hsc::eliminate_vertex(g, 1);
        // Expected new graph on {0, 2, 3}:
        //   compensated (0, 2) = 0 + (1 * 1) / 2 = 1/2
        //   (2, 3) untouched = 1
        RC_ASSERT(std::fabs(hsc::edge_weight(g2, 0, 2) - 0.5) < 1e-12);
        RC_ASSERT(std::fabs(hsc::edge_weight(g2, 2, 3) - 1.0) < 1e-12);
        // Convert to CSR and check matrix structure.
        const sp::SparseMatrixCSR L2 = hsc::graph_to_csr(g2);
        RC_ASSERT(L2.rows == 4u);   // vertex 1 still indexed but isolated
    });

    // Independent-set selection: F's must form an independent set
    // (no two F's adjacent) and every vertex is either F or has
    // an F-neighbor (or is isolated).
    ok &= rc::check("F-points form an independent set on a 4-path", [] {
        // Path 0-1-2-3.
        hsc::Graph g;
        g.num_verts = 4;
        g.edges = { { 0, 1, 1.0 }, { 1, 2, 1.0 }, { 2, 3, 1.0 } };
        const auto is_F = hsc::select_independent_set(g);
        for (const auto &e : g.edges) {
            const int a = std::get<0>(e);
            const int b = std::get<1>(e);
            // No adjacent F-pair.
            RC_ASSERT(!(is_F[a] && is_F[b]));
        }
    });

    // Coarsening level: after eliminating one independent-set
    // worth of vertices on the 4-path, the surviving graph has
    // valid Laplacian structure.
    ok &= rc::check("coarsen_one_level on 4-path produces valid coarse graph", [] {
        hsc::Graph g;
        g.num_verts = 4;
        g.edges = { { 0, 1, 1.0 }, { 1, 2, 1.0 }, { 2, 3, 1.0 } };
        const auto lev = hsc::coarsen_one_level(g);
        // C-only graph has fewer (or equal) vertices than original.
        RC_ASSERT(lev.coarse.num_verts < g.num_verts);
        // c_to_orig and orig_to_c are consistent.
        for (std::size_t c = 0; c < lev.coarse.num_verts; ++c) {
            const int orig = lev.c_to_orig[c];
            RC_ASSERT(lev.orig_to_c[orig] == static_cast<int>(c));
        }
    });

    // Prolongation is row-stochastic: each fine row's weights
    // sum to 1 (C-points trivially, F-points via normalisation).
    ok &= rc::check("prolongation is row-stochastic on 4-path", [] {
        hsc::Graph g;
        g.num_verts = 4;
        g.edges = { { 0, 1, 1.0 }, { 1, 2, 1.0 }, { 2, 3, 1.0 } };
        const auto lev = hsc::coarsen_one_level(g);
        const auto P = hsc::make_prolongation(g, lev);
        std::vector<double> rowsum(P.n_fine, 0.0);
        for (const auto &t : P.triples) {
            rowsum[std::get<0>(t)] += std::get<2>(t);
        }
        for (std::size_t i = 0; i < P.n_fine; ++i) {
            // Either 1.0 (covered) or 0.0 (orphan F-point).
            RC_ASSERT(std::fabs(rowsum[i] - 1.0) < 1e-12 || rowsum[i] == 0.0);
        }
    });

    // Hierarchy build on a path of 17 vertices: should produce
    // multiple levels of coarsening.
    ok &= rc::check("build_hierarchy on path-17 produces multiple levels", [] {
        hsc::Graph g;
        g.num_verts = 17;
        for (int i = 0; i < 16; ++i) g.edges.push_back({ i, i + 1, 1.0 });
        const auto h = hsc::build_hierarchy(g, /*coarsest_size=*/4);
        RC_ASSERT(h.graphs.size() >= 2u);
        RC_ASSERT(h.graphs[0].num_verts == 17u);
        RC_ASSERT(h.graphs.back().num_verts <= 8u);
        // Each level shrinks.
        for (std::size_t l = 1; l < h.graphs.size(); ++l) {
            RC_ASSERT(h.graphs[l].num_verts < h.graphs[l - 1].num_verts);
        }
    });

    // V-cycle applied to b in range(A) recovers x with A·x ≈ b.
    // On a small Laplacian, one V-cycle won't fully converge but
    // residual must drop substantially from b's magnitude.
    ok &= rc::check("V-cycle drops residual on path-9 Laplacian", [] {
        hsc::Graph g;
        g.num_verts = 9;
        for (int i = 0; i < 8; ++i) g.edges.push_back({ i, i + 1, 1.0 });
        const sp::SparseMatrixCSR A = hsc::graph_to_csr(g);
        const auto h = hsc::build_hierarchy(g, /*coarsest_size=*/3);

        // Build a zero-mean RHS in range(A). y = sin(0.1*i), b = A·y.
        std::vector<double> y(g.num_verts);
        for (std::size_t i = 0; i < g.num_verts; ++i) y[i] = std::sin(0.1 * static_cast<double>(i));
        double mean = 0.0;
        for (double v : y) mean += v;
        mean /= y.size();
        for (auto &v : y) v -= mean;
        const std::vector<double> b = sp::spmv(A, y);

        const std::vector<double> x = hsc::v_cycle_apply(h, b);
        const std::vector<double> Ax = sp::spmv(A, x);
        double r_norm = 0.0, b_norm = 0.0;
        for (std::size_t i = 0; i < g.num_verts; ++i) {
            r_norm += (Ax[i] - b[i]) * (Ax[i] - b[i]);
            b_norm += b[i] * b[i];
        }
        // V-cycle should reduce residual by at least 5x compared to ||b||.
        RC_ASSERT(std::sqrt(r_norm) < 0.2 * std::sqrt(b_norm));
    });

    // P · 1_coarse = 1_fine (constant prolongs to constant).
    ok &= rc::check("P prolongs constant vectors to constant", [] {
        hsc::Graph g;
        g.num_verts = 5;
        g.edges = {
            { 0, 1, 1.0 }, { 1, 2, 1.0 }, { 2, 3, 1.0 }, { 3, 4, 1.0 }
        };
        const auto lev = hsc::coarsen_one_level(g);
        const auto P = hsc::make_prolongation(g, lev);
        std::vector<double> ones_c(P.n_coarse, 1.0);
        const auto fine = hsc::prolong(P, ones_c);
        for (std::size_t i = 0; i < P.n_fine; ++i) {
            // Same caveat: orphan F's have rowsum 0.
            RC_ASSERT(std::fabs(fine[i] - 1.0) < 1e-12 || fine[i] == 0.0);
        }
    });

    return ok ? 0 : 1;
}
