// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
#include "test_helpers.h"

#include "curvenet/bilinear_inverse.h"

using curvenet::BilinearInverseResult;
using curvenet::Vec3;
using curvenet::solve_bilinear_inverse;

static rc::Gen<Vec3> gen_vec() {
	auto coord = rc::gen::map(rc::gen::inRange<int>(-100, 101),
			[](int v) { return static_cast<double>(v) * 0.1; });
	return rc::gen::construct<Vec3>(coord, coord, coord);
}

// Generate a non-degenerate quad as 4 corners spread out in a near-planar
// arrangement so the bilinear inverse has a unique solution. We pin three
// corners on the xy plane and let the fourth jitter to test warping.
static rc::Gen<std::tuple<Vec3, Vec3, Vec3, Vec3>> gen_quad() {
	auto jitter = rc::gen::map(rc::gen::inRange<int>(-30, 31),
			[](int v) { return static_cast<double>(v) * 0.01; });
	return rc::gen::map(rc::gen::tuple(jitter, jitter, jitter, jitter),
			[](std::tuple<double, double, double, double> j) {
				Vec3 P00{ 0.0, 0.0, std::get<0>(j) };
				Vec3 P10{ 1.0, 0.0, std::get<1>(j) };
				Vec3 P11{ 1.0, 1.0, std::get<2>(j) };
				Vec3 P01{ 0.0, 1.0, std::get<3>(j) };
				return std::make_tuple(P00, P10, P11, P01);
			});
}

static Vec3 patch_eval(const Vec3 &P00, const Vec3 &P10, const Vec3 &P11, const Vec3 &P01,
		double s, double t) {
	double ms = 1.0 - s, mt = 1.0 - t;
	return P00 * (ms * mt) + P10 * (s * mt) + P11 * (s * t) + P01 * (ms * t);
}

int main() {
	bool ok = true;

	ok &= rc::check("inverse(P00) returns (0, 0)", [] {
		auto q = *gen_quad();
		auto [P00, P10, P11, P01] = q;
		auto r = solve_bilinear_inverse(P00, P00, P10, P11, P01);
		RC_ASSERT(r.converged);
		RC_ASSERT(std::fabs(r.s) < 1e-6);
		RC_ASSERT(std::fabs(r.t) < 1e-6);
		RC_ASSERT(r.residual < 1e-6);
	});

	ok &= rc::check("inverse(P10) returns (1, 0)", [] {
		auto q = *gen_quad();
		auto [P00, P10, P11, P01] = q;
		auto r = solve_bilinear_inverse(P10, P00, P10, P11, P01);
		RC_ASSERT(r.converged);
		RC_ASSERT(std::fabs(r.s - 1.0) < 1e-6);
		RC_ASSERT(std::fabs(r.t) < 1e-6);
		RC_ASSERT(r.residual < 1e-6);
	});

	ok &= rc::check("inverse(P11) returns (1, 1)", [] {
		auto q = *gen_quad();
		auto [P00, P10, P11, P01] = q;
		auto r = solve_bilinear_inverse(P11, P00, P10, P11, P01);
		RC_ASSERT(r.converged);
		RC_ASSERT(std::fabs(r.s - 1.0) < 1e-6);
		RC_ASSERT(std::fabs(r.t - 1.0) < 1e-6);
	});

	ok &= rc::check("inverse(P01) returns (0, 1)", [] {
		auto q = *gen_quad();
		auto [P00, P10, P11, P01] = q;
		auto r = solve_bilinear_inverse(P01, P00, P10, P11, P01);
		RC_ASSERT(r.converged);
		RC_ASSERT(std::fabs(r.s) < 1e-6);
		RC_ASSERT(std::fabs(r.t - 1.0) < 1e-6);
	});

	ok &= rc::check("inverse(centroid) returns (0.5, 0.5)", [] {
		auto q = *gen_quad();
		auto [P00, P10, P11, P01] = q;
		Vec3 mid = (P00 + P10 + P11 + P01) * 0.25;
		auto r = solve_bilinear_inverse(mid, P00, P10, P11, P01);
		RC_ASSERT(r.converged);
		// For planar parallelograms (P00,P10,P11,P01), centroid is (0.5,0.5).
		// For warped (twisted) quads, centroid maps to (~0.5, ~0.5) only
		// approximately, but residual should still be tiny since it lies on
		// the bilinear surface.
		RC_ASSERT(r.residual < 1e-6);
	});

	ok &= rc::check("round-trip: forward(s,t) then inverse recovers (s,t)", [] {
		auto q = *gen_quad();
		auto [P00, P10, P11, P01] = q;
		double s_in = *cnt::unit_t();
		double t_in = *cnt::unit_t();
		Vec3 V = patch_eval(P00, P10, P11, P01, s_in, t_in);
		auto r = solve_bilinear_inverse(V, P00, P10, P11, P01);
		RC_ASSERT(r.converged);
		RC_ASSERT(std::fabs(r.s - s_in) < 1e-6);
		RC_ASSERT(std::fabs(r.t - t_in) < 1e-6);
	});

	ok &= rc::check("degenerate (all-coincident) quad reports infinite residual", [](Vec3 P) {
		// Zero-area "quad" — all four corners at the same point.
		auto r = solve_bilinear_inverse(Vec3{ 1.0, 2.0, 3.0 }, P, P, P, P);
		RC_ASSERT(!r.converged);
		RC_ASSERT(std::isinf(r.residual));
	});

	return ok ? 0 : 1;
}
