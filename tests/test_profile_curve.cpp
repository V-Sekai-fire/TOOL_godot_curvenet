// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
#include "test_helpers.h"

#include "curvenet/profile_curve.h"

using curvenet::ProfileCurve;
using curvenet::Vec3;

// Generate a closed profile curve with 3..6 handles and matching tangents.
static rc::Gen<ProfileCurve> gen_profile() {
	return rc::gen::mapcat(rc::gen::inRange<int>(3, 7), [](int n) {
		auto coord = rc::gen::map(rc::gen::inRange<int>(-100, 101),
				[](int v) { return static_cast<double>(v) * 0.1; });
		auto vec = rc::gen::construct<Vec3>(coord, coord, coord);
		auto vecs = rc::gen::container<std::vector<Vec3>>(n, vec);
		return rc::gen::map(rc::gen::tuple(vecs, vecs, vecs),
				[](std::tuple<std::vector<Vec3>, std::vector<Vec3>, std::vector<Vec3>> t) {
					ProfileCurve c;
					c.handles = std::get<0>(t);
					c.tangents_out = std::get<1>(t);
					c.tangents_in = std::get<2>(t);
					return c;
				});
	});
}

int main() {
	bool ok = true;

	ok &= rc::check("profile.evaluate(0) returns handles[0]", [] {
		ProfileCurve c = *gen_profile();
		Vec3 r = c.evaluate(0.0);
		RC_ASSERT(cnt::approx_eq(r, c.handles[0]));
	});

	ok &= rc::check("profile.evaluate(1) wraps to handles[0] (closed loop)", [] {
		ProfileCurve c = *gen_profile();
		Vec3 r = c.evaluate(1.0);
		RC_ASSERT(cnt::approx_eq(r, c.handles[0]));
	});

	ok &= rc::check("profile.evaluate(k/N) returns handles[k]", [] {
		ProfileCurve c = *gen_profile();
		int n = c.num_segments();
		int k = *rc::gen::inRange<int>(0, n);
		Vec3 r = c.evaluate(static_cast<double>(k) / n);
		RC_ASSERT(cnt::approx_eq(r, c.handles[k], 1e-7));
	});

	ok &= rc::check("constant profile evaluates to constant point", [](Vec3 p) {
		int n = *rc::gen::inRange<int>(3, 7);
		ProfileCurve c;
		c.handles.assign(n, p);
		c.tangents_out.assign(n, p);
		c.tangents_in.assign(n, p);
		double t = *cnt::unit_t();
		Vec3 r = c.evaluate(t);
		RC_ASSERT(cnt::approx_eq(r, p, 1e-9));
	});

	ok &= rc::check("translating all controls translates evaluation", [](Vec3 d) {
		ProfileCurve c = *gen_profile();
		ProfileCurve c2 = c;
		for (auto &h : c2.handles) h = h + d;
		for (auto &h : c2.tangents_out) h = h + d;
		for (auto &h : c2.tangents_in) h = h + d;
		double t = *cnt::unit_t();
		Vec3 a = c.evaluate(t) + d;
		Vec3 b = c2.evaluate(t);
		RC_ASSERT(cnt::approx_eq(a, b, 1e-9));
	});

	ok &= rc::check("tangent at segment start equals 3*(tangents_out[k] - handles[k])", [] {
		ProfileCurve c = *gen_profile();
		int n = c.num_segments();
		int k = *rc::gen::inRange<int>(0, n);
		Vec3 r = c.evaluate_tangent(static_cast<double>(k) / n);
		// Profile derivative: d(P(s/N))/ds = (1/N) * dB/dlocal_t * N = dB/dlocal_t.
		// At local_t=0, dB/dlocal_t = 3*(p1 - p0) = 3*(tangents_out[k] - handles[k]).
		// But wait — global derivative dP/dt = N * dB/dlocal_t.
		Vec3 expected = (c.tangents_out[k] - c.handles[k]) * 3.0 * n;
		RC_ASSERT(cnt::approx_eq(r, expected, 1e-7));
	});

	return ok ? 0 : 1;
}
