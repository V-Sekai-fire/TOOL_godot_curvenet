// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
//
// TOMBSTONE [loop 100/2, 2026-05-09]
// Tested for the wrong reason. The 81k matrix's constant kernel is
// already exact (max |row_sum| = 1.16e-10) and `b` lies in range to
// ~1e-13, so zero-mean projection has nothing to clean up. Adding
// it to the V-cycle was harmless on both 5k and 81k.
// See tests/diag_70k_cg_baseline.cpp and PERF_BASELINE.md
// "Dead ends". Kept as infrastructure; do not expect it to
// accelerate convergence on its own.
//
// ----
//
// C++ mirror of `lean/Curvenet/KernelProjection.lean`.
//
// Project a vector onto the orthogonal complement of the constant
// mode (1, 1, ..., 1)^T, i.e. subtract its mean. Used after every
// V-cycle correction to keep the multilevel solver from drifting
// in the Laplacian's 1D constant null-space.

#ifndef CURVENET_KERNEL_PROJECTION_H
#define CURVENET_KERNEL_PROJECTION_H

#include <cstddef>
#include <vector>

namespace curvenet {
namespace kernel_projection {

inline void zero_mean_in_place(std::vector<double> &v) {
    const std::size_t n = v.size();
    if (n == 0) return;
    double s = 0.0;
    for (double x : v) s += x;
    const double m = s / static_cast<double>(n);
    for (double &x : v) x -= m;
}

inline std::vector<double> zero_mean(const std::vector<double> &v) {
    std::vector<double> out = v;
    zero_mean_in_place(out);
    return out;
}

inline double mean(const std::vector<double> &v) {
    if (v.empty()) return 0.0;
    double s = 0.0;
    for (double x : v) s += x;
    return s / static_cast<double>(v.size());
}

} // namespace kernel_projection
} // namespace curvenet

#endif
