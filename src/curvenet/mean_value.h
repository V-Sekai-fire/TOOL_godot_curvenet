// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
#ifndef CURVENET_MEAN_VALUE_H
#define CURVENET_MEAN_VALUE_H

#include <cmath>
#include <vector>

namespace curvenet {

// Mean-value coordinates (Floater 2003) for an N-sided convex polygon, used as
// the parametric domain of an N>=5 patch. Mirrors lean/Curvenet/MeanValue.lean.
//
// Tangent half-angle form (no atan2 needed):
//   d_k       = V_k - p
//   r_k       = |d_k|
//   tau_k     = cross(d_k, d_{k+1}) / (r_k * r_{k+1} + dot(d_k, d_{k+1}))
//                                        -- equals tan(alpha_k / 2) for
//                                           alpha_k = angle(V_k, p, V_{k+1})
//   raw_k     = (tau_{k-1} + tau_k) / r_k
//   lambda_k  = raw_k / sum_j raw_j
//
// At a polygon vertex r_k = 0, so we special-case: if `p` bit-equals any V_k,
// return the indicator vector (1.0 at k, 0.0 elsewhere).

struct Vec2 {
	double x = 0.0;
	double y = 0.0;

	Vec2() = default;
	Vec2(double x_, double y_) : x(x_), y(y_) {}

	Vec2 operator-(const Vec2 &o) const { return { x - o.x, y - o.y }; }

	double norm() const { return std::sqrt(x * x + y * y); }
	double dot(const Vec2 &o) const { return x * o.x + y * o.y; }
	double cross(const Vec2 &o) const { return x * o.y - y * o.x; }

	bool operator==(const Vec2 &o) const { return x == o.x && y == o.y; }
};

// Regular N-gon on the unit circle, vertex k at angle 2*pi*k/N.
inline std::vector<Vec2> regular_ngon(int n) {
	const double two_pi = 6.283185307179586;
	std::vector<Vec2> out;
	out.reserve(n);
	for (int k = 0; k < n; ++k) {
		double theta = two_pi * k / static_cast<double>(n);
		out.push_back({ std::cos(theta), std::sin(theta) });
	}
	return out;
}

// Mean-value barycentric coordinates of `p` against polygon `poly` (CCW).
// Length of returned vector matches poly.size().
inline std::vector<double> mean_value_weights(const std::vector<Vec2> &poly, const Vec2 &p) {
	const int n = static_cast<int>(poly.size());
	std::vector<double> out(n, 0.0);
	if (n == 0) {
		return out;
	}

	// Vertex coincidence: indicator vector.
	for (int k = 0; k < n; ++k) {
		if (poly[k] == p) {
			out[k] = 1.0;
			return out;
		}
	}

	std::vector<Vec2> d(n);
	std::vector<double> r(n);
	for (int k = 0; k < n; ++k) {
		d[k] = poly[k] - p;
		r[k] = d[k].norm();
	}
	std::vector<double> tan_half(n);
	for (int k = 0; k < n; ++k) {
		const int kp = (k + 1) % n;
		double c = d[k].cross(d[kp]);
		double dt = d[k].dot(d[kp]);
		double denom = r[k] * r[kp] + dt;
		// denom == 0 means p sits on the ray opposite to the side -- defer to
		// the unnormalized form; standard MVC stays well-defined for convex
		// interiors so this branch should not trigger in practice.
		tan_half[k] = denom == 0.0 ? 0.0 : c / denom;
	}
	std::vector<double> raw(n);
	double total = 0.0;
	for (int k = 0; k < n; ++k) {
		const int km = (k + n - 1) % n;
		raw[k] = (tan_half[km] + tan_half[k]) / r[k];
		total += raw[k];
	}
	if (total == 0.0) {
		return out;
	}
	for (int k = 0; k < n; ++k) {
		out[k] = raw[k] / total;
	}
	return out;
}

} // namespace curvenet

#endif
