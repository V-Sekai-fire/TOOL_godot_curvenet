// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
#ifndef CURVENET_INTERSECTION_FRAMES_H
#define CURVENET_INTERSECTION_FRAMES_H

// C++ mirror of `lean/Curvenet/IntersectionFrames.lean`. Curvenet
// intersection geometry: corner normals m_i, per-side normals n_i^±,
// per-side widths w_i^±, and the assembled per-side scaled frame
// matrices B_i S_i. Includes the §3 T-junction degenerate fallback for
// parallel consecutive tangents.

#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "scaled_frames.h"
#include "vec3.h"

namespace curvenet {
namespace intersection_frames {

struct OutgoingSegment {
	Vec3   tangent;
	double length = 0.0;
};

inline Vec3 cross(const Vec3 &a, const Vec3 &b) {
	return { a.y * b.z - a.z * b.y,
	         a.z * b.x - a.x * b.z,
	         a.x * b.y - a.y * b.x };
}

inline std::vector<Vec3> corner_vectors(const std::vector<OutgoingSegment> &segs) {
	const std::size_t n = segs.size();
	std::vector<Vec3> acc(n, Vec3{ 0.0, 0.0, 0.0 });
	for (std::size_t i = 0; i < n; ++i) {
		acc[i] = cross(segs[i].tangent, segs[(i + 1) % n].tangent);
	}
	return acc;
}

constexpr double kDegenerateEps = 1.0e-12;

// Corner normals m_i = c_i / |c_i| with the §3 T-junction fallback
// m_i = (c_{i+1} + c_{i-1}) / |c_{i+1} + c_{i-1}| when |c_i| ≤ eps.
// Returns the zero vector at an index iff the fallback denominator is
// also zero (triple-collinear configuration).
inline std::vector<Vec3> corner_normals(const std::vector<OutgoingSegment> &segs) {
	const std::vector<Vec3> cs = corner_vectors(segs);
	const std::size_t n = cs.size();
	std::vector<Vec3> acc(n, Vec3{ 0.0, 0.0, 0.0 });
	for (std::size_t i = 0; i < n; ++i) {
		const Vec3 c = cs[i];
		const double nrm = c.length();
		if (nrm > kDegenerateEps) {
			const double inv = 1.0 / nrm;
			acc[i] = { c.x * inv, c.y * inv, c.z * inv };
		} else {
			const Vec3 cp = cs[(i + 1) % n];
			const Vec3 cm = cs[(i + n - 1) % n];
			const Vec3 sum = { cp.x + cm.x, cp.y + cm.y, cp.z + cm.z };
			const double s_norm = sum.length();
			if (s_norm > kDegenerateEps) {
				const double inv = 1.0 / s_norm;
				acc[i] = { sum.x * inv, sum.y * inv, sum.z * inv };
			}
			// else leave zero
		}
	}
	return acc;
}

// Per-side normals: (n_i^+, n_i^-) = (m_i, m_{i-1}).
inline std::vector<std::pair<Vec3, Vec3>>
per_side_normals(const std::vector<OutgoingSegment> &segs) {
	const std::vector<Vec3> ms = corner_normals(segs);
	const std::size_t n = ms.size();
	std::vector<std::pair<Vec3, Vec3>> acc(n, { Vec3{ 0.0, 0.0, 0.0 }, Vec3{ 0.0, 0.0, 0.0 } });
	for (std::size_t i = 0; i < n; ++i) {
		acc[i] = { ms[i], ms[(i + n - 1) % n] };
	}
	return acc;
}

// Per-side widths:
//   w_i^+ = l_i + |c_i|     · (l_{i+1} − l_i)
//   w_i^- = l_i + |c_{i-1}| · (l_{i-1} − l_i)
inline std::vector<std::pair<double, double>>
per_side_widths(const std::vector<OutgoingSegment> &segs) {
	const std::vector<Vec3> cs = corner_vectors(segs);
	const std::size_t n = segs.size();
	std::vector<std::pair<double, double>> acc(n, { 0.0, 0.0 });
	for (std::size_t i = 0; i < n; ++i) {
		const double li  = segs[i].length;
		const double lip = segs[(i + 1) % n].length;
		const double lim = segs[(i + n - 1) % n].length;
		const double c_mag  = cs[i].length();
		const double c_magm = cs[(i + n - 1) % n].length();
		const double w_plus  = li + c_mag  * (lip - li);
		const double w_minus = li + c_magm * (lim - li);
		acc[i] = { w_plus, w_minus };
	}
	return acc;
}

// Per-side scaled frame B S = [t·l | b·w | n·h] with b = n × t and
// h = sqrt(l · w). Mirrors slice 14.
inline scaled_frames::Mat3 scaled_frame(const Vec3 &t, const Vec3 &n, double l, double w) {
	const double bin_x = n.y * t.z - n.z * t.y;
	const double bin_y = n.z * t.x - n.x * t.z;
	const double bin_z = n.x * t.y - n.y * t.x;
	const double h = std::sqrt(l * w);
	return scaled_frames::mat3_make(
		t.x * l, bin_x * w, n.x * h,
		t.y * l, bin_y * w, n.y * h,
		t.z * l, bin_z * w, n.z * h);
}

inline std::vector<std::pair<scaled_frames::Mat3, scaled_frames::Mat3>>
per_side_scaled_frames(const std::vector<OutgoingSegment> &segs) {
	const std::size_t n = segs.size();
	const auto normals = per_side_normals(segs);
	const auto widths  = per_side_widths(segs);
	std::vector<std::pair<scaled_frames::Mat3, scaled_frames::Mat3>> acc;
	acc.reserve(n);
	for (std::size_t i = 0; i < n; ++i) {
		const Vec3 t = segs[i].tangent;
		const double l = segs[i].length;
		const Vec3 np = normals[i].first;
		const Vec3 nm = normals[i].second;
		const double wp = widths[i].first;
		const double wm = widths[i].second;
		acc.push_back({ scaled_frame(t, np, l, wp), scaled_frame(t, nm, l, wm) });
	}
	return acc;
}

} // namespace intersection_frames
} // namespace curvenet

#endif
