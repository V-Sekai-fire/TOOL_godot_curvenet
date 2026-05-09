// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
#include "test_helpers.h"

#include "curvenet/bezier.h"

using curvenet::Vec3;
using curvenet::evaluate_cubic_bezier;
using curvenet::evaluate_cubic_bezier_derivative;

int main() {
	bool ok = true;

	ok &= rc::check("bezier endpoint at t=0 equals p0",
			[](Vec3 p0, Vec3 p1, Vec3 p2, Vec3 p3) {
				Vec3 r = evaluate_cubic_bezier(p0, p1, p2, p3, 0.0);
				RC_ASSERT(cnt::approx_eq(r, p0));
			});

	ok &= rc::check("bezier endpoint at t=1 equals p3",
			[](Vec3 p0, Vec3 p1, Vec3 p2, Vec3 p3) {
				Vec3 r = evaluate_cubic_bezier(p0, p1, p2, p3, 1.0);
				RC_ASSERT(cnt::approx_eq(r, p3));
			});

	ok &= rc::check("bezier with 4 coincident control points is constant",
			[](Vec3 p) {
				double t = *cnt::unit_t();
				Vec3 r = evaluate_cubic_bezier(p, p, p, p, t);
				RC_ASSERT(cnt::approx_eq(r, p));
			});

	ok &= rc::check("bezier basis sums to 1 (affine invariance under translation)",
			[](Vec3 p0, Vec3 p1, Vec3 p2, Vec3 p3, Vec3 d) {
				double t = *cnt::unit_t();
				Vec3 a = evaluate_cubic_bezier(p0, p1, p2, p3, t) + d;
				Vec3 b = evaluate_cubic_bezier(p0 + d, p1 + d, p2 + d, p3 + d, t);
				RC_ASSERT(cnt::approx_eq(a, b, 1e-9));
			});

	ok &= rc::check("derivative at t=0 equals 3*(p1-p0)",
			[](Vec3 p0, Vec3 p1, Vec3 p2, Vec3 p3) {
				Vec3 r = evaluate_cubic_bezier_derivative(p0, p1, p2, p3, 0.0);
				RC_ASSERT(cnt::approx_eq(r, (p1 - p0) * 3.0));
			});

	ok &= rc::check("derivative at t=1 equals 3*(p3-p2)",
			[](Vec3 p0, Vec3 p1, Vec3 p2, Vec3 p3) {
				Vec3 r = evaluate_cubic_bezier_derivative(p0, p1, p2, p3, 1.0);
				RC_ASSERT(cnt::approx_eq(r, (p3 - p2) * 3.0));
			});

	return ok ? 0 : 1;
}
