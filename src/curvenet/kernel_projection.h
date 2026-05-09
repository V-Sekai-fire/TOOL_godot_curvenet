// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
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
