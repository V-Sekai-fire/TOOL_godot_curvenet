// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
#include "test_helpers.h"

#include "curvenet/mean_value.h"

#include <cmath>

using curvenet::Vec2;
using curvenet::mean_value_weights;
using curvenet::regular_ngon;

int main() {
	bool ok = true;

	ok &= rc::check("vertex Lagrange: lambda_k(V_k) = 1, others = 0", [] {
		int n = *rc::gen::inRange<int>(3, 9);
		auto poly = regular_ngon(n);
		int k = *rc::gen::inRange<int>(0, n);
		auto w = mean_value_weights(poly, poly[k]);
		RC_ASSERT(static_cast<int>(w.size()) == n);
		for (int i = 0; i < n; ++i) {
			double expected = (i == k) ? 1.0 : 0.0;
			RC_ASSERT(std::fabs(w[i] - expected) < 1e-12);
		}
	});

	ok &= rc::check("partition of unity at the centroid", [] {
		int n = *rc::gen::inRange<int>(3, 9);
		auto poly = regular_ngon(n);
		auto w = mean_value_weights(poly, Vec2{ 0.0, 0.0 });
		double s = 0.0;
		for (double x : w) {
			s += x;
		}
		RC_ASSERT(std::fabs(s - 1.0) < 1e-12);
	});

	ok &= rc::check("symmetry: regular N-gon centroid gives uniform 1/N weights", [] {
		int n = *rc::gen::inRange<int>(3, 9);
		auto poly = regular_ngon(n);
		auto w = mean_value_weights(poly, Vec2{ 0.0, 0.0 });
		double inv_n = 1.0 / n;
		for (double x : w) {
			RC_ASSERT(std::fabs(x - inv_n) < 1e-12);
		}
	});

	ok &= rc::check("partition of unity at arbitrary interior points", [] {
		int n = *rc::gen::inRange<int>(3, 9);
		auto poly = regular_ngon(n);
		// Sample inside a small disk around origin (well inside the convex hull).
		double r_in = *rc::gen::map(rc::gen::inRange<int>(0, 50),
				[](int v) { return static_cast<double>(v) * 0.01; }); // 0..0.5
		double theta = *rc::gen::map(rc::gen::inRange<int>(0, 360),
				[](int v) { return static_cast<double>(v) * 0.0174533; });
		Vec2 p{ r_in * std::cos(theta), r_in * std::sin(theta) };
		auto w = mean_value_weights(poly, p);
		double s = 0.0;
		for (double x : w) {
			s += x;
		}
		RC_ASSERT(std::fabs(s - 1.0) < 1e-9);
	});

	return ok ? 0 : 1;
}
