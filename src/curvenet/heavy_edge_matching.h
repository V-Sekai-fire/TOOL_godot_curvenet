// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
//
// TOMBSTONE [loop 100/1, 2026-05-09]
// Does not fix the 81k V-cycle stall. Connectivity-aware HEM
// produced the same residual plateau (~3.7) as principal-axis
// bucketing. Aggregation is not the bottleneck. See
// tests/diag_70k_cg_baseline.cpp and PERF_BASELINE.md "Dead ends".
// Header is correct + tested; reuse with a different smoother.
//
// ----
//
// C++ mirror of `lean/Curvenet/HeavyEdgeMatching.lean`.
//
// Heavy-edge matching (Karypis-Kumar 1998) — connectivity-aware
// 2:1 pair-aggregation. One pass walks node ids in order, pairs
// each unmatched node with the first unmatched neighbor, leaves
// the rest as singletons. Iterating gives 2^k:1 coarsening with
// every aggregate guaranteed connected.

#ifndef CURVENET_HEAVY_EDGE_MATCHING_H
#define CURVENET_HEAVY_EDGE_MATCHING_H

#include <algorithm>
#include <cstddef>
#include <unordered_set>
#include <vector>

namespace curvenet {
namespace heavy_edge_matching {

using Adjacency = std::vector<std::vector<int>>;

struct MatchResult {
    std::vector<int> cmap;       // length = adj.size(), values in [0, num_coarse)
    std::size_t      num_coarse; // number of clusters produced
};

// One pass of HEM: dense cluster ids in [0, num_coarse).
inline MatchResult hem_match(const Adjacency &adj) {
    const std::size_t n = adj.size();
    std::vector<int>  cmap(n, 0);
    std::vector<char> matched(n, 0);
    std::size_t next_id = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (matched[i]) continue;
        bool paired = false;
        for (int j : adj[i]) {
            if (j == static_cast<int>(i)) continue;
            if (matched[j]) continue;
            cmap[i] = static_cast<int>(next_id);
            cmap[j] = static_cast<int>(next_id);
            matched[i] = matched[j] = 1;
            ++next_id;
            paired = true;
            break;
        }
        if (!paired) {
            cmap[i] = static_cast<int>(next_id);
            matched[i] = 1;
            ++next_id;
        }
    }
    return { std::move(cmap), next_id };
}

// Iterate `hem_match` k times, composing the cmaps. Each iteration
// also coarsens the adjacency graph so the next pass operates on
// the cluster-level adjacency.
inline MatchResult iterate_hem(const Adjacency &adj0, std::size_t k) {
    const std::size_t n0 = adj0.size();
    std::vector<int> cmap(n0);
    for (std::size_t i = 0; i < n0; ++i) cmap[i] = static_cast<int>(i);
    Adjacency adj = adj0;
    std::size_t num_c = n0;
    for (std::size_t step_i = 0; step_i < k; ++step_i) {
        const MatchResult step = hem_match(adj);
        for (std::size_t i = 0; i < n0; ++i) {
            cmap[i] = step.cmap[cmap[i]];
        }
        // Build coarsened adjacency.
        std::vector<std::unordered_set<int>> sets(step.num_coarse);
        for (std::size_t i = 0; i < adj.size(); ++i) {
            const int ci = step.cmap[i];
            for (int j : adj[i]) {
                const int cj = step.cmap[j];
                if (ci != cj) sets[ci].insert(cj);
            }
        }
        Adjacency next_adj(step.num_coarse);
        for (std::size_t c = 0; c < step.num_coarse; ++c) {
            next_adj[c].assign(sets[c].begin(), sets[c].end());
            std::sort(next_adj[c].begin(), next_adj[c].end());
        }
        adj = std::move(next_adj);
        num_c = step.num_coarse;
    }
    return { std::move(cmap), num_c };
}

// Iterate HEM until num_coarse <= target_max. Returns the cumulative
// cmap from original fine indices to final cluster ids.
inline MatchResult coarsen_until(const Adjacency &adj0, std::size_t target_max) {
    if (adj0.size() <= target_max) {
        std::vector<int> identity(adj0.size());
        for (std::size_t i = 0; i < adj0.size(); ++i) identity[i] = static_cast<int>(i);
        return { std::move(identity), adj0.size() };
    }
    std::size_t k = 0;
    std::size_t cur = adj0.size();
    while (cur > target_max) {
        cur = (cur + 1) / 2;   // upper bound on size after one HEM pass
        ++k;
        if (k > 64) break;     // safety
    }
    return iterate_hem(adj0, k);
}

} // namespace heavy_edge_matching
} // namespace curvenet

#endif
