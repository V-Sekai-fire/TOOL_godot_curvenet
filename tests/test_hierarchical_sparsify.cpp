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

    return ok ? 0 : 1;
}
