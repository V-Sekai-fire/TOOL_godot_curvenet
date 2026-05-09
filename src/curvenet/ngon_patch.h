// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
#ifndef CURVENET_NGON_PATCH_H
#define CURVENET_NGON_PATCH_H

#include "coons_patch.h"
#include "mean_value.h"
#include "vec3.h"

#include <cmath>
#include <vector>

namespace curvenet {

// N-sided patch bounded by N cubic Bezier curves. Adjacent boundaries share a corner:
// boundaries[i].c3 == boundaries[(i+1) % N].c0 (CCW orientation).
//
// Evaluation domain is a regular N-gon. A query point is given in barycentric-style
// "side parameters" (s, t) only meaningful for N=3 and N=4 in this cycle:
//   - N=4: (s, t) in [0,1]^2, mapped through CoonsPatch.
//   - N=3: (s, t) in [0,1]^2 with implicit third coord (1 - s - t), uses a
//     degenerate Coons (top side collapsed to corner V2).
//   - N>=5: TODO — mean-value-coords blending (Hormann-Floater style). Returns NaN.
struct NgonPatch {
	std::vector<BoundaryCurve> boundaries;

	int n() const { return static_cast<int>(boundaries.size()); }

	Vec3 evaluate(double s, double t) const {
		const int sides = n();
		if (sides == 4) {
			// CCW boundary loop: bottom (b0), right (b1), top reversed (b2), left reversed (b3).
			// CoonsPatch wants u0/u1 in canonical u direction and v0/v1 in canonical v direction.
			CoonsPatch c;
			c.u0 = boundaries[0]; // P00 -> P10
			c.v1 = boundaries[1]; // P10 -> P11
			// boundaries[2] runs P11 -> P01; reverse to get canonical u1 (P01 -> P11).
			c.u1 = BoundaryCurve{ boundaries[2].c3, boundaries[2].c2, boundaries[2].c1, boundaries[2].c0 };
			// boundaries[3] runs P01 -> P00; reverse to get canonical v0 (P00 -> P01).
			c.v0 = BoundaryCurve{ boundaries[3].c3, boundaries[3].c2, boundaries[3].c1, boundaries[3].c0 };
			return c.evaluate(s, t);
		}
		if (sides == 3) {
			// Degenerate Coons: triangle (P0, P1, P2) with CCW edges
			//   boundaries[0] = P0->P1, boundaries[1] = P1->P2, boundaries[2] = P2->P0.
			// Map (s, t) ∈ [0, 1]² to the triangle so the v=1 edge collapses to P2:
			//   u0(u) = boundaries[0]                        (P0 -> P1, bottom, v=0)
			//   v1(v) = boundaries[1]                        (P1 -> P2, right, u=1)
			//   v0(v) = reverse(boundaries[2])               (P0 -> P2, left, u=0)
			//   u1(u) = constant P2                          (top edge collapsed)
			const Vec3 P2 = boundaries[1].c3; // = boundaries[2].c0
			CoonsPatch c;
			c.u0 = boundaries[0];
			c.v1 = boundaries[1];
			c.v0 = BoundaryCurve{ boundaries[2].c3, boundaries[2].c2, boundaries[2].c1, boundaries[2].c0 };
			c.u1 = BoundaryCurve{ P2, P2, P2, P2 };
			return c.evaluate(s, t);
		}
		// N >= 5: blend N boundary vertex positions by mean-value coords on a
		// regular N-gon domain. (s, t) is the 2D query point inside the unit-
		// radius regular N-gon; vertex i of the patch is the start of side i.
		// Gives partition-of-unity translation invariance and exact recovery
		// at each domain vertex. Boundary recovery along sides is approximate;
		// extending to side-MVC for exact transfinite is future work.
		std::vector<Vec2> domain = regular_ngon(sides);
		std::vector<double> w = mean_value_weights(domain, Vec2{ s, t });
		Vec3 acc{ 0.0, 0.0, 0.0 };
		for (int i = 0; i < sides; ++i) {
			acc += boundaries[i].c0 * w[i];
		}
		return acc;
	}
};

} // namespace curvenet

#endif
