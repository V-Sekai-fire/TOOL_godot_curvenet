// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
//
// RapidCheck mirror of HeavyEdgeMatching.lean's native_decide
// proofs.

#include "curvenet/heavy_edge_matching.h"

#include <rapidcheck.h>
#include <vector>

namespace hem = curvenet::heavy_edge_matching;

int main() {
    bool ok = true;

    ok &= rc::check("empty 4-node graph: every node is a singleton", [] {
        const hem::Adjacency adj = { {}, {}, {}, {} };
        const auto r = hem::hem_match(adj);
        RC_ASSERT(r.cmap.size() == 4u);
        RC_ASSERT(r.num_coarse == 4u);
        RC_ASSERT(r.cmap == std::vector<int>({ 0, 1, 2, 3 }));
    });

    ok &= rc::check("path 0-1-2-3 -> 2 clusters", [] {
        const hem::Adjacency adj = { { 1 }, { 0, 2 }, { 1, 3 }, { 2 } };
        const auto r = hem::hem_match(adj);
        RC_ASSERT(r.num_coarse == 2u);
        RC_ASSERT(r.cmap == std::vector<int>({ 0, 0, 1, 1 }));
    });

    ok &= rc::check("star (0 connected to 1,2,3): 0 pairs with 1, leaves are singletons", [] {
        const hem::Adjacency adj = { { 1, 2, 3 }, { 0 }, { 0 }, { 0 } };
        const auto r = hem::hem_match(adj);
        RC_ASSERT(r.num_coarse == 3u);
        RC_ASSERT(r.cmap == std::vector<int>({ 0, 0, 1, 2 }));
    });

    ok &= rc::check("4-cycle -> 2 clusters", [] {
        const hem::Adjacency adj = { { 1, 3 }, { 0, 2 }, { 1, 3 }, { 0, 2 } };
        const auto r = hem::hem_match(adj);
        RC_ASSERT(r.num_coarse == 2u);
        RC_ASSERT(r.cmap == std::vector<int>({ 0, 0, 1, 1 }));
    });

    ok &= rc::check("path 0..7: 1 pass yields 4 clusters", [] {
        const hem::Adjacency adj = {
            { 1 }, { 0, 2 }, { 1, 3 }, { 2, 4 },
            { 3, 5 }, { 4, 6 }, { 5, 7 }, { 6 }
        };
        const auto r = hem::iterate_hem(adj, 1);
        RC_ASSERT(r.num_coarse == 4u);
        RC_ASSERT(r.cmap == std::vector<int>({ 0, 0, 1, 1, 2, 2, 3, 3 }));
    });

    ok &= rc::check("path 0..7: 2 passes yield 2 clusters", [] {
        const hem::Adjacency adj = {
            { 1 }, { 0, 2 }, { 1, 3 }, { 2, 4 },
            { 3, 5 }, { 4, 6 }, { 5, 7 }, { 6 }
        };
        const auto r = hem::iterate_hem(adj, 2);
        RC_ASSERT(r.num_coarse == 2u);
        RC_ASSERT(r.cmap == std::vector<int>({ 0, 0, 0, 0, 1, 1, 1, 1 }));
    });

    ok &= rc::check("coarsen_until target=2 on path 0..7: matches 2-pass result", [] {
        const hem::Adjacency adj = {
            { 1 }, { 0, 2 }, { 1, 3 }, { 2, 4 },
            { 3, 5 }, { 4, 6 }, { 5, 7 }, { 6 }
        };
        const auto r = hem::coarsen_until(adj, 2);
        RC_ASSERT(r.num_coarse <= 2u);
    });

    return ok ? 0 : 1;
}
