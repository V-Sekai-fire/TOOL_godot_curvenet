// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
//
// Verifies `curvenet_builder` ε-merges shared endpoints across curves
// and classifies each merged knot per DeGoes22 §3 (anchor / regular /
// intersection). Models the topology in Fig. 1 of the panda-character
// paper (multiple curves with intersections) at small scale.
#include "test_helpers.h"

#include "curvenet/curvenet_builder.h"

#include <vector>

using curvenet::Vec3;
namespace cb = curvenet::curvenet_builder;

int main() {
	bool ok = true;

	// --- Case A: a single open curve with 3 knots (a—b—c). ---
	ok &= rc::check("single open curve: 3 knots, knot 1 regular, 0/2 anchors", [] {
		const std::vector<cb::CurvePoints> curves = {
			{ { 0.0, 0.0, 0.0 }, { 1.0, 0.0, 0.0 }, { 2.0, 0.0, 0.0 } } };
		const cb::CurvenetGraph g = cb::build(curves, 1e-9);
		RC_ASSERT(g.knot_positions.size() == 3);
		const std::vector<cb::KnotKind> kinds = cb::classify(g);
		RC_ASSERT(kinds[0] == cb::KnotKind::anchor);
		RC_ASSERT(kinds[1] == cb::KnotKind::regular);
		RC_ASSERT(kinds[2] == cb::KnotKind::anchor);
	});

	// --- Case B: two curves sharing an endpoint (Y branch but only 2
	//     curves, so the join is just regular — 2 incident splines). ---
	ok &= rc::check("Y-join of 2 curves: 3 unique knots, junction is regular", [] {
		const std::vector<cb::CurvePoints> curves = {
			{ { 0.0, 0.0, 0.0 }, { 1.0, 0.0, 0.0 } },   // curve 0
			{ { 1.0, 0.0, 0.0 }, { 2.0, 1.0, 0.0 } } }; // curve 1, shares (1,0,0)
		const cb::CurvenetGraph g = cb::build(curves, 1e-9);
		RC_ASSERT(g.knot_positions.size() == 3);
		const std::vector<cb::KnotKind> kinds = cb::classify(g);
		// Knot at (1, 0, 0) has 2 incident segments (one per curve) → regular.
		// Find it by position.
		std::size_t shared = 0;
		for (std::size_t i = 0; i < g.knot_positions.size(); ++i) {
			const Vec3 &p = g.knot_positions[i];
			if (p.x == 1.0 && p.y == 0.0 && p.z == 0.0) {
				shared = i;
				break;
			}
		}
		RC_ASSERT(kinds[shared] == cb::KnotKind::regular);
	});

	// --- Case C: 3 curves emanating from a center (T-junction-like). ---
	ok &= rc::check("3 curves from one center: center is intersection", [] {
		const std::vector<cb::CurvePoints> curves = {
			{ { 0.0, 0.0, 0.0 }, {  1.0,  0.0, 0.0 } },   // east
			{ { 0.0, 0.0, 0.0 }, { -1.0,  0.0, 0.0 } },   // west
			{ { 0.0, 0.0, 0.0 }, {  0.0,  1.0, 0.0 } } }; // north
		const cb::CurvenetGraph g = cb::build(curves, 1e-9);
		RC_ASSERT(g.knot_positions.size() == 4); // center + 3 tips
		const std::vector<cb::KnotKind> kinds = cb::classify(g);
		// Find the center knot (the one with 3 incidences).
		std::size_t center = 0;
		for (std::size_t i = 0; i < g.knot_positions.size(); ++i) {
			if (g.incidence[i].size() == 3) {
				center = i;
				break;
			}
		}
		RC_ASSERT(kinds[center] == cb::KnotKind::intersection);
	});

	// --- Case D: outgoing tangents at a 3-curve intersection point along
	//     +x, +y, -x; expected 3 unit vectors away from the centre. ---
	ok &= rc::check("3-curve intersection: 3 outgoing unit tangents", [] {
		const std::vector<cb::CurvePoints> curves = {
			{ { 0.0, 0.0, 0.0 }, {  1.0, 0.0, 0.0 } },
			{ { 0.0, 0.0, 0.0 }, {  0.0, 1.0, 0.0 } },
			{ { 0.0, 0.0, 0.0 }, { -1.0, 0.0, 0.0 } } };
		const cb::CurvenetGraph g = cb::build(curves, 1e-9);
		const auto tangents = cb::outgoing_tangents(g);
		std::size_t center = 0;
		for (std::size_t i = 0; i < g.knot_positions.size(); ++i) {
			if (g.incidence[i].size() == 3) {
				center = i;
				break;
			}
		}
		RC_ASSERT(tangents[center].size() == 3);
		// All three should be unit vectors.
		for (const Vec3 &t : tangents[center]) {
			RC_ASSERT(std::abs(t.length() - 1.0) < 1e-12);
		}
	});

	// --- Case E: ε-merging tolerates small noise. Two curves with knots
	//     within ε but not exactly equal still merge. ---
	ok &= rc::check("epsilon merging absorbs sub-eps noise", [] {
		const std::vector<cb::CurvePoints> curves = {
			{ { 0.0, 0.0, 0.0 }, { 1.0, 0.0, 0.0 } },
			{ { 1.0 + 1e-10, 0.0, 0.0 }, { 2.0, 0.0, 0.0 } } };
		const cb::CurvenetGraph g = cb::build(curves, 1e-6);
		// Expect 3 unique knots (the two near-(1,0,0) merged).
		RC_ASSERT(g.knot_positions.size() == 3);
	});

	// --- Case F: closed loop (last knot duplicated as first).
	//     For the 4-handle Curve3D pattern in the demo. ---
	ok &= rc::check("closed-loop curve: ε-merge identifies wraparound", [] {
		const std::vector<cb::CurvePoints> curves = {
			{ { -1.0, 0.4, -1.0 },
			  {  1.0, 0.4, -1.0 },
			  {  1.0, 0.4,  1.0 },
			  { -1.0, 0.4,  1.0 },
			  { -1.0, 0.4, -1.0 } } };
		const cb::CurvenetGraph g = cb::build(curves, 1e-9);
		// 4 unique knots (last == first merged).
		RC_ASSERT(g.knot_positions.size() == 4);
		// Each knot is incident to exactly 2 segments (closed loop).
		const std::vector<cb::KnotKind> kinds = cb::classify(g);
		for (cb::KnotKind k : kinds) {
			RC_ASSERT(k == cb::KnotKind::regular);
		}
	});

	return ok ? 0 : 1;
}
