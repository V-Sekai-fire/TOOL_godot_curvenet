// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
//
// Mirrors the §3 frames layer instance theorems from Lean (slices 4,
// 13, 14, 15, 16, 18). Uses `test_helpers.h` for RapidCheck wiring.
#include "test_helpers.h"

#include "curvenet/curve_interp.h"
#include "curvenet/intersection_frames.h"
#include "curvenet/scaled_frames.h"
#include "curvenet/segment_gradient.h"

#include <cmath>
#include <vector>

using curvenet::Vec3;
namespace sf = curvenet::scaled_frames;
namespace ifr = curvenet::intersection_frames;
namespace ci = curvenet::curve_interp;
namespace sg = curvenet::segment_gradient;

int main() {
	bool ok = true;

	const Vec3 origin = { 0.0, 0.0, 0.0 };
	const Vec3 x_unit = { 1.0, 0.0, 0.0 };
	const Vec3 y_unit = { 0.0, 1.0, 0.0 };

	// --- ScaledFrames slice 4 ---

	ok &= rc::check("isolated F: rest = posed -> identity", [origin, x_unit] {
		const sf::Mat3 F = sf::isolated_segment_gradient(origin, x_unit, origin, x_unit);
		RC_ASSERT(sf::mat3_within_eps(F, sf::mat3_identity(), 1e-12));
	});

	ok &= rc::check("isolated F: pure rotation = R_90", [origin, x_unit, y_unit] {
		const sf::Mat3 F = sf::isolated_segment_gradient(origin, x_unit, origin, y_unit);
		const sf::Mat3 R90 = sf::mat3_make(0.0, -1.0, 0.0,
		                                    1.0,  0.0, 0.0,
		                                    0.0,  0.0, 1.0);
		RC_ASSERT(sf::mat3_within_eps(F, R90, 1e-12));
	});

	// --- ScaledFrames slice 15 ---

	ok &= rc::check("det(I) = 1", [] {
		RC_ASSERT(std::abs(sf::mat3_det(sf::mat3_identity()) - 1.0) < 1e-12);
	});

	ok &= rc::check("inv(diag) round-trip", [] {
		const sf::Mat3 M = sf::mat3_make(2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0);
		const sf::Mat3 round = sf::mat3_mul(sf::mat3_inv(M), M);
		RC_ASSERT(sf::mat3_within_eps(round, sf::mat3_identity(), 1e-10));
	});

	ok &= rc::check("deformation_gradient: rest = posed -> I", [] {
		const sf::Mat3 M = sf::mat3_make(2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0);
		const sf::Mat3 F = sf::deformation_gradient(M, M);
		RC_ASSERT(sf::mat3_within_eps(F, sf::mat3_identity(), 1e-10));
	});

	// --- IntersectionFrames slice 13–14 ---

	ok &= rc::check("perpendicular intersection: corner normals = +/-z", [] {
		const std::vector<ifr::OutgoingSegment> segs = {
			{ { 1.0, 0.0, 0.0 }, 1.0 },
			{ { 0.0, 1.0, 0.0 }, 1.0 } };
		const std::vector<Vec3> ms = ifr::corner_normals(segs);
		RC_ASSERT(std::abs(ms[0].z - 1.0) < 1e-12 &&
		           std::abs(ms[1].z + 1.0) < 1e-12);
	});

	ok &= rc::check("scaled_frame: identity inputs -> identity matrix", [] {
		const sf::Mat3 M = ifr::scaled_frame({ 1.0, 0.0, 0.0 },
		                                      { 0.0, 0.0, 1.0 }, 1.0, 1.0);
		RC_ASSERT(sf::mat3_within_eps(M, sf::mat3_identity(), 1e-12));
	});

	// --- IntersectionFrames slice 17 (T-junction fallback) ---

	ok &= rc::check("T-junction fallback: anti-parallel pair recovers +z", [] {
		const std::vector<ifr::OutgoingSegment> tj = {
			{ {  1.0, 0.0, 0.0 }, 1.0 },
			{ {  0.0, 1.0, 0.0 }, 1.0 },
			{ { -1.0, 0.0, 0.0 }, 1.0 } };
		const std::vector<Vec3> ms = ifr::corner_normals(tj);
		RC_ASSERT(std::abs(ms[2].z - 1.0) < 1e-12);
		// And the recovered normal is unit-length.
		RC_ASSERT(std::abs(ms[2].length() - 1.0) < 1e-12);
	});

	// --- CurveInterp slice 16 ---

	ok &= rc::check("interp_along_curve: 3 segments midpoint blends 50/50", [] {
		const std::pair<ci::SideData, ci::SideData> first = {
			{ { 0.0, 0.0, 1.0 }, 1.0 }, { { 0.0, 0.0, -1.0 }, 1.0 } };
		const std::pair<ci::SideData, ci::SideData> last = {
			{ { 1.0, 0.0, 0.0 }, 5.0 }, { { -1.0, 0.0, 0.0 }, 3.0 } };
		const auto arr = ci::interp_along_curve(first, last, 3);
		const ci::SideData mid_p = arr[1].first;
		RC_ASSERT(std::abs(mid_p.n.x - 0.5) < 1e-12 &&
		           std::abs(mid_p.n.z - 0.5) < 1e-12 &&
		           std::abs(mid_p.w - 3.0) < 1e-12);
	});

	// --- SegmentGradient slice 18 ---

	ok &= rc::check("dispatcher: isolated rest = posed -> I", [origin, x_unit] {
		const sf::Mat3 F = sg::isolated(origin, x_unit, origin, x_unit);
		RC_ASSERT(sf::mat3_within_eps(F, sf::mat3_identity(), 1e-12));
	});

	ok &= rc::check("dispatcher: intersection rest = posed -> I per side", [] {
		const sf::Mat3 frame = sf::mat3_identity();
		const auto pair = sg::intersection_pair(frame, frame, frame, frame);
		RC_ASSERT(sf::mat3_within_eps(pair.first,  sf::mat3_identity(), 1e-12) &&
		           sf::mat3_within_eps(pair.second, sf::mat3_identity(), 1e-12));
	});

	return ok ? 0 : 1;
}
