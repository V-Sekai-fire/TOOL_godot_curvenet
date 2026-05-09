// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
#ifndef CURVENET_COONS_PATCH_H
#define CURVENET_COONS_PATCH_H

#include "bezier.h"
#include "vec3.h"

namespace curvenet {

// One side of a Coons patch — a cubic Bezier curve.
struct BoundaryCurve {
	Vec3 c0;
	Vec3 c1;
	Vec3 c2;
	Vec3 c3;

	Vec3 evaluate(double s) const { return evaluate_cubic_bezier(c0, c1, c2, c3, s); }
};

// Bilinear-blended Coons patch over 4 cubic boundary curves.
// Naming convention:
//   u0(u): bottom edge, v = 0, varies in u from corner (0,0) to (1,0).
//   u1(u): top edge,    v = 1, varies in u from corner (0,1) to (1,1).
//   v0(v): left edge,   u = 0, varies in v from corner (0,0) to (0,1).
//   v1(v): right edge,  u = 1, varies in v from corner (1,0) to (1,1).
//
// Corner compatibility: u0.c0 == v0.c0, u0.c3 == v1.c0, u1.c0 == v0.c3, u1.c3 == v1.c3.
struct CoonsPatch {
	BoundaryCurve u0;
	BoundaryCurve u1;
	BoundaryCurve v0;
	BoundaryCurve v1;

	// S(u,v) = Lc(u,v) + Ld(u,v) - B(u,v), where
	//   Lc = (1-v) * u0(u)  + v * u1(u)             — ruled in v
	//   Ld = (1-u) * v0(v)  + u * v1(v)             — ruled in u
	//   B  = bilinear blend of the four corners
	// This is the classical bilinear-blended Coons patch.
	Vec3 evaluate(double u, double v) const {
		const double mu = 1.0 - u;
		const double mv = 1.0 - v;

		const Vec3 cu0 = u0.evaluate(u);
		const Vec3 cu1 = u1.evaluate(u);
		const Vec3 cv0 = v0.evaluate(v);
		const Vec3 cv1 = v1.evaluate(v);

		const Vec3 Lc = cu0 * mv + cu1 * v;
		const Vec3 Ld = cv0 * mu + cv1 * u;

		const Vec3 P00 = u0.c0;
		const Vec3 P10 = u0.c3;
		const Vec3 P01 = u1.c0;
		const Vec3 P11 = u1.c3;
		const Vec3 B = P00 * (mu * mv) + P10 * (u * mv) + P01 * (mu * v) + P11 * (u * v);

		return Lc + Ld - B;
	}
};

} // namespace curvenet

#endif
