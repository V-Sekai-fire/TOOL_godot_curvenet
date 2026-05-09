// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
#ifndef CURVENET_BEZIER_H
#define CURVENET_BEZIER_H

#include "vec3.h"

namespace curvenet {

inline Vec3 evaluate_cubic_bezier(const Vec3 &p0, const Vec3 &p1, const Vec3 &p2, const Vec3 &p3, double t) {
	const double s = 1.0 - t;
	const double b0 = s * s * s;
	const double b1 = 3.0 * s * s * t;
	const double b2 = 3.0 * s * t * t;
	const double b3 = t * t * t;
	return p0 * b0 + p1 * b1 + p2 * b2 + p3 * b3;
}

inline Vec3 evaluate_cubic_bezier_derivative(const Vec3 &p0, const Vec3 &p1, const Vec3 &p2, const Vec3 &p3, double t) {
	const double s = 1.0 - t;
	const double d0 = 3.0 * s * s;
	const double d1 = 6.0 * s * t;
	const double d2 = 3.0 * t * t;
	return (p1 - p0) * d0 + (p2 - p1) * d1 + (p3 - p2) * d2;
}

} // namespace curvenet

#endif
