// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
#include "test_helpers.h"

#include "curvenet/ngon_patch.h"

using curvenet::BoundaryCurve;
using curvenet::CoonsPatch;
using curvenet::NgonPatch;
using curvenet::Vec3;

static rc::Gen<Vec3> gen_vec() {
	auto coord = rc::gen::map(rc::gen::inRange<int>(-100, 101),
			[](int v) { return static_cast<double>(v) * 0.1; });
	return rc::gen::construct<Vec3>(coord, coord, coord);
}

// Build a 4-sided NgonPatch with corner-consistent boundaries (analogous to gen_patch in test_coons_patch).
static rc::Gen<std::pair<NgonPatch, CoonsPatch>> gen_quad_pair() {
	return rc::gen::map(
			rc::gen::tuple(gen_vec(), gen_vec(), gen_vec(), gen_vec(),
					gen_vec(), gen_vec(),
					gen_vec(), gen_vec(),
					gen_vec(), gen_vec(),
					gen_vec(), gen_vec()),
			[](std::tuple<Vec3, Vec3, Vec3, Vec3, Vec3, Vec3, Vec3, Vec3, Vec3, Vec3, Vec3, Vec3> t) {
				const Vec3 P00 = std::get<0>(t);
				const Vec3 P10 = std::get<1>(t);
				const Vec3 P11 = std::get<2>(t);
				const Vec3 P01 = std::get<3>(t);
				CoonsPatch c;
				c.u0 = BoundaryCurve{ P00, std::get<4>(t), std::get<5>(t), P10 };
				c.u1 = BoundaryCurve{ P01, std::get<6>(t), std::get<7>(t), P11 };
				c.v0 = BoundaryCurve{ P00, std::get<8>(t), std::get<9>(t), P01 };
				c.v1 = BoundaryCurve{ P10, std::get<10>(t), std::get<11>(t), P11 };
				// CCW boundary loop for the n-gon: bottom (P00→P10), right (P10→P11), top reversed (P11→P01), left reversed (P01→P00).
				NgonPatch ng;
				ng.boundaries.push_back(c.u0); // bottom: P00 -> P10
				ng.boundaries.push_back(c.v1); // right:  P10 -> P11
				BoundaryCurve top_rev{ c.u1.c3, c.u1.c2, c.u1.c1, c.u1.c0 };
				ng.boundaries.push_back(top_rev); // top reversed: P11 -> P01
				BoundaryCurve left_rev{ c.v0.c3, c.v0.c2, c.v0.c1, c.v0.c0 };
				ng.boundaries.push_back(left_rev); // left reversed: P01 -> P00
				return std::make_pair(ng, c);
			});
}

int main() {
	bool ok = true;

	ok &= rc::check("4-sided NgonPatch corners interpolate", [] {
		auto pair = *gen_quad_pair();
		const NgonPatch &ng = pair.first;
		// Corners of the unit square in (s, t).
		RC_ASSERT(cnt::approx_eq(ng.evaluate(0.0, 0.0), ng.boundaries[0].c0, 1e-9));
		RC_ASSERT(cnt::approx_eq(ng.evaluate(1.0, 0.0), ng.boundaries[0].c3, 1e-9));
		RC_ASSERT(cnt::approx_eq(ng.evaluate(1.0, 1.0), ng.boundaries[1].c3, 1e-9));
		RC_ASSERT(cnt::approx_eq(ng.evaluate(0.0, 1.0), ng.boundaries[3].c0, 1e-9));
	});

	ok &= rc::check("4-sided NgonPatch matches CoonsPatch", [] {
		auto pair = *gen_quad_pair();
		const NgonPatch &ng = pair.first;
		const CoonsPatch &c = pair.second;
		double s = *cnt::unit_t();
		double t = *cnt::unit_t();
		RC_ASSERT(cnt::approx_eq(ng.evaluate(s, t), c.evaluate(s, t), 1e-9));
	});

	ok &= rc::check("4-sided NgonPatch translation invariance", [](Vec3 d) {
		auto pair = *gen_quad_pair();
		NgonPatch ng = pair.first;
		NgonPatch ng2 = ng;
		for (auto &b : ng2.boundaries) {
			b.c0 = b.c0 + d;
			b.c1 = b.c1 + d;
			b.c2 = b.c2 + d;
			b.c3 = b.c3 + d;
		}
		double s = *cnt::unit_t();
		double t = *cnt::unit_t();
		RC_ASSERT(cnt::approx_eq(ng.evaluate(s, t) + d, ng2.evaluate(s, t), 1e-9));
	});

	return ok ? 0 : 1;
}
