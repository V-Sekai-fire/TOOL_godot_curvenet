// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
//
// TOMBSTONE [loop 100/2, 2026-05-09]
// Does not fix the 81k V-cycle stall; same residual plateau (~3.7)
// as 1-, 2-, 3-, and 7-level variants. Root cause is the 7-decade
// diagonal spread, not level count or aggregator. See
// tests/diag_70k_cg_baseline.cpp and PERF_BASELINE.md "Dead ends".
// Header is correct + tested; reuse with a smoother robust to
// wide-range D before re-trying on any large mesh.
//
// ----
//
// C++ mirror of `lean/Curvenet/MultiLevelSchwarz.lean`. Recursive
// extension of `two_level_schwarz.h` — instead of one coarse level,
// stacks several aggregations so the effective coarsening goes from
// fine size to a handful of nodes at the top, where direct solve
// is trivial.
//
// Why we need this: two-level Schwarz at 81k stalled in 427 outer
// iters because the 256:1 single aggregation produced a 368-node
// coarse problem with similar conditioning to the fine grid. Wu
// 2022's recursive 32:1 hierarchy (e.g. 81613 → 2550 → 80 → 3)
// shrinks the coarsest problem to a few rows, where direct solve
// is exact and converges the bulk of the global modes per outer
// iter.

#ifndef CURVENET_MULTI_LEVEL_SCHWARZ_H
#define CURVENET_MULTI_LEVEL_SCHWARZ_H

#include <cstddef>
#include <vector>

#include "two_level_schwarz.h"

namespace curvenet {
namespace multi_level_schwarz {

// A `Hierarchy` is a sequence of CoarseMaps (one per level
// transition). `cmaps[i]` maps level-i indices to level-(i+1)
// cluster ids; `num_coarse_at[i]` is the number of nodes at
// level i (so `num_coarse_at[0]` is the fine size and
// `num_coarse_at.back()` is the top size).
struct Hierarchy {
    std::vector<std::vector<int>> cmaps;          // length = numLevels - 1
    std::vector<std::size_t>      num_coarse_at;  // length = numLevels
};

// Apply the chain of restrictions to send a fine vector to the
// coarsest level.
inline std::vector<double> restrict_through(const Hierarchy &h,
                                                const std::vector<double> &f) {
    std::vector<double> v = f;
    for (std::size_t i = 0; i < h.cmaps.size(); ++i) {
        v = two_level_schwarz::restrict_fine(h.cmaps[i],
                                                h.num_coarse_at[i + 1], v);
    }
    return v;
}

// Apply the chain of prolongations from coarsest back to fine.
inline std::vector<double> prolong_through(const Hierarchy &h,
                                              const std::vector<double> &c) {
    std::vector<double> v = c;
    const std::size_t n_levels = h.cmaps.size();
    for (std::size_t j = 0; j < n_levels; ++j) {
        const std::size_t i = n_levels - 1 - j;
        v = two_level_schwarz::prolong_coarse(h.cmaps[i],
                                                 h.num_coarse_at[i], v);
    }
    return v;
}

inline std::size_t fine_size(const Hierarchy &h) {
    return h.num_coarse_at.empty() ? 0 : h.num_coarse_at.front();
}

inline std::size_t top_size(const Hierarchy &h) {
    return h.num_coarse_at.empty() ? 0 : h.num_coarse_at.back();
}

inline std::size_t num_levels(const Hierarchy &h) {
    return h.num_coarse_at.size();
}

} // namespace multi_level_schwarz
} // namespace curvenet

#endif
