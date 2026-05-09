// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
#ifndef CURVENET_CURVE_INTERP_H
#define CURVENET_CURVE_INTERP_H

// C++ mirror of `lean/Curvenet/CurveInterp.lean`. Linear blend of
// per-side (normal, width) values along a curve between two
// intersection-supplied endpoints.

#include <cstddef>
#include <utility>
#include <vector>

#include "vec3.h"

namespace curvenet {
namespace curve_interp {

struct SideData {
	Vec3   n;
	double w = 0.0;
};

inline Vec3 lerp_vec3(const Vec3 &a, const Vec3 &b, double t) {
	return { (1.0 - t) * a.x + t * b.x,
	         (1.0 - t) * a.y + t * b.y,
	         (1.0 - t) * a.z + t * b.z };
}

inline double lerp_float(double a, double b, double t) {
	return (1.0 - t) * a + t * b;
}

inline SideData lerp_side(const SideData &a, const SideData &b, double t) {
	return { lerp_vec3(a.n, b.n, t), lerp_float(a.w, b.w, t) };
}

// Linear interpolation along a curve over `n` segments. Index 0 carries
// `first`, index n-1 carries `last`; intermediate indices are blended.
inline std::vector<std::pair<SideData, SideData>>
interp_along_curve(const std::pair<SideData, SideData> &first,
                    const std::pair<SideData, SideData> &last,
                    std::size_t n) {
	if (n == 0) {
		return {};
	}
	if (n == 1) {
		return { first };
	}
	std::vector<std::pair<SideData, SideData>> acc(n, first);
	const double denom = static_cast<double>(n - 1);
	for (std::size_t i = 0; i < n; ++i) {
		const double t = static_cast<double>(i) / denom;
		acc[i] = { lerp_side(first.first,  last.first,  t),
		           lerp_side(first.second, last.second, t) };
	}
	return acc;
}

} // namespace curve_interp
} // namespace curvenet

#endif
