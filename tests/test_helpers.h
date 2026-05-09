// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
#ifndef CURVENET_TEST_HELPERS_H
#define CURVENET_TEST_HELPERS_H

#include <rapidcheck.h>

#include <algorithm>
#include <cmath>

#include "curvenet/vec3.h"

namespace rc {
template <>
struct Arbitrary<curvenet::Vec3> {
	static Gen<curvenet::Vec3> arbitrary() {
		auto coord = gen::map(gen::inRange<int>(-1000, 1001),
				[](int v) { return static_cast<double>(v) * 0.01; });
		return gen::construct<curvenet::Vec3>(coord, coord, coord);
	}
};
} // namespace rc

namespace cnt {

inline bool approx_eq(double a, double b, double eps = 1e-9) {
	return std::fabs(a - b) <= eps * std::max(1.0, std::max(std::fabs(a), std::fabs(b)));
}

inline bool approx_eq(const curvenet::Vec3 &a, const curvenet::Vec3 &b, double eps = 1e-9) {
	return approx_eq(a.x, b.x, eps) && approx_eq(a.y, b.y, eps) && approx_eq(a.z, b.z, eps);
}

inline auto unit_t() {
	return rc::gen::map(rc::gen::inRange<int>(0, 1001),
			[](int v) { return static_cast<double>(v) * 0.001; });
}

} // namespace cnt

#endif
