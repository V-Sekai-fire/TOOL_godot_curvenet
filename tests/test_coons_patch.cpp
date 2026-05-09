// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
#include "test_helpers.h"

#include "curvenet/coons_patch.h"

using curvenet::BoundaryCurve;
using curvenet::CoonsPatch;
using curvenet::Vec3;

// Build a corner-consistent Coons patch from 4 corners and 8 free tangent control points.
static rc::Gen<CoonsPatch> gen_patch() {
	auto coord = rc::gen::map(rc::gen::inRange<int>(-100, 101),
			[](int v) { return static_cast<double>(v) * 0.1; });
	auto vec = rc::gen::construct<Vec3>(coord, coord, coord);
	return rc::gen::map(
			rc::gen::tuple(vec, vec, vec, vec, // 4 corners P00, P10, P01, P11
					vec, vec, // u0 inner controls (between P00 and P10)
					vec, vec, // u1 inner controls (between P01 and P11)
					vec, vec, // v0 inner controls (between P00 and P01)
					vec, vec), // v1 inner controls (between P10 and P11)
			[](std::tuple<Vec3, Vec3, Vec3, Vec3, Vec3, Vec3, Vec3, Vec3, Vec3, Vec3, Vec3, Vec3> t) {
				CoonsPatch p;
				const Vec3 P00 = std::get<0>(t);
				const Vec3 P10 = std::get<1>(t);
				const Vec3 P01 = std::get<2>(t);
				const Vec3 P11 = std::get<3>(t);
				p.u0 = BoundaryCurve{ P00, std::get<4>(t), std::get<5>(t), P10 };
				p.u1 = BoundaryCurve{ P01, std::get<6>(t), std::get<7>(t), P11 };
				p.v0 = BoundaryCurve{ P00, std::get<8>(t), std::get<9>(t), P01 };
				p.v1 = BoundaryCurve{ P10, std::get<10>(t), std::get<11>(t), P11 };
				return p;
			});
}

int main() {
	bool ok = true;

	ok &= rc::check("S(u, 0) recovers bottom boundary u0(u)", [] {
		CoonsPatch p = *gen_patch();
		double u = *cnt::unit_t();
		Vec3 a = p.evaluate(u, 0.0);
		Vec3 b = p.u0.evaluate(u);
		RC_ASSERT(cnt::approx_eq(a, b, 1e-9));
	});

	ok &= rc::check("S(u, 1) recovers top boundary u1(u)", [] {
		CoonsPatch p = *gen_patch();
		double u = *cnt::unit_t();
		Vec3 a = p.evaluate(u, 1.0);
		Vec3 b = p.u1.evaluate(u);
		RC_ASSERT(cnt::approx_eq(a, b, 1e-9));
	});

	ok &= rc::check("S(0, v) recovers left boundary v0(v)", [] {
		CoonsPatch p = *gen_patch();
		double v = *cnt::unit_t();
		Vec3 a = p.evaluate(0.0, v);
		Vec3 b = p.v0.evaluate(v);
		RC_ASSERT(cnt::approx_eq(a, b, 1e-9));
	});

	ok &= rc::check("S(1, v) recovers right boundary v1(v)", [] {
		CoonsPatch p = *gen_patch();
		double v = *cnt::unit_t();
		Vec3 a = p.evaluate(1.0, v);
		Vec3 b = p.v1.evaluate(v);
		RC_ASSERT(cnt::approx_eq(a, b, 1e-9));
	});

	ok &= rc::check("corners interpolate", [] {
		CoonsPatch p = *gen_patch();
		RC_ASSERT(cnt::approx_eq(p.evaluate(0.0, 0.0), p.u0.c0, 1e-9));
		RC_ASSERT(cnt::approx_eq(p.evaluate(1.0, 0.0), p.u0.c3, 1e-9));
		RC_ASSERT(cnt::approx_eq(p.evaluate(0.0, 1.0), p.u1.c0, 1e-9));
		RC_ASSERT(cnt::approx_eq(p.evaluate(1.0, 1.0), p.u1.c3, 1e-9));
	});

	ok &= rc::check("translation of all controls translates evaluation", [](Vec3 d) {
		CoonsPatch p = *gen_patch();
		CoonsPatch q = p;
		auto shift = [&d](BoundaryCurve &b) {
			b.c0 = b.c0 + d;
			b.c1 = b.c1 + d;
			b.c2 = b.c2 + d;
			b.c3 = b.c3 + d;
		};
		shift(q.u0);
		shift(q.u1);
		shift(q.v0);
		shift(q.v1);
		double u = *cnt::unit_t();
		double v = *cnt::unit_t();
		Vec3 a = p.evaluate(u, v) + d;
		Vec3 b = q.evaluate(u, v);
		RC_ASSERT(cnt::approx_eq(a, b, 1e-9));
	});

	return ok ? 0 : 1;
}
