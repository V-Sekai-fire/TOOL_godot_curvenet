// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
#ifndef CURVENET_PROFILE_CURVE_H
#define CURVENET_PROFILE_CURVE_H

#include "bezier.h"
#include "vec3.h"

#include <cmath>
#include <vector>

namespace curvenet {

// Closed piecewise-cubic-Bezier loop.
// `handles[i]` are the through-points (knots) shared by adjacent segments.
// `tangents_out[i]` is the outgoing tangent control point at handle i (absolute position, not offset).
// `tangents_in[i]`  is the incoming tangent control point at handle i.
// Segment i runs handle[i] -> tangent_out[i] -> tangent_in[(i+1)%N] -> handle[(i+1)%N].
// Per-handle in/out are independent so creases are representable.
struct ProfileCurve {
	std::vector<Vec3> handles;
	std::vector<Vec3> tangents_out;
	std::vector<Vec3> tangents_in;

	int num_segments() const { return static_cast<int>(handles.size()); }

	// Map global t in [0,1] to a segment index k and local parameter u in [0,1].
	// Returns the four control points of segment k.
	void locate(double t, int &out_k, double &out_u, Vec3 &p0, Vec3 &p1, Vec3 &p2, Vec3 &p3) const {
		const int n = num_segments();
		double scaled = t * n;
		int k = static_cast<int>(std::floor(scaled));
		// Clamp left and right (right boundary maps to last segment with u=1, which then wraps to handles[0]).
		if (k < 0) {
			k = 0;
		}
		if (k >= n) {
			k = n - 1;
		}
		double u = scaled - static_cast<double>(k);
		// Numerical safety: if t==1.0 exactly, scaled==n, k clamped to n-1, u becomes 1.0.
		const int next = (k + 1) % n;
		out_k = k;
		out_u = u;
		p0 = handles[k];
		p1 = tangents_out[k];
		p2 = tangents_in[next];
		p3 = handles[next];
	}

	Vec3 evaluate(double t) const {
		int k;
		double u;
		Vec3 p0, p1, p2, p3;
		locate(t, k, u, p0, p1, p2, p3);
		return evaluate_cubic_bezier(p0, p1, p2, p3, u);
	}

	// Global tangent dP/dt = N * dB/du via chain rule.
	Vec3 evaluate_tangent(double t) const {
		int k;
		double u;
		Vec3 p0, p1, p2, p3;
		locate(t, k, u, p0, p1, p2, p3);
		return evaluate_cubic_bezier_derivative(p0, p1, p2, p3, u) * static_cast<double>(num_segments());
	}
};

// Godot-style curve handle: position + relative incoming/outgoing tangent offsets.
// Mirrors Godot's Curve3D (point_position, point_in, point_out where point_in/out
// are stored relative to the point position). Kept here so cycle 6 binding code
// can convert without pulling godot-cpp into the math layer.
struct CurveHandle {
	Vec3 position;
	Vec3 in_offset;
	Vec3 out_offset;
};

// Build a closed ProfileCurve from a list of CurveHandles (Godot Curve3D conventions).
inline ProfileCurve profile_from_handles(const std::vector<CurveHandle> &handles) {
	ProfileCurve c;
	c.handles.reserve(handles.size());
	c.tangents_in.reserve(handles.size());
	c.tangents_out.reserve(handles.size());
	for (const auto &h : handles) {
		c.handles.push_back(h.position);
		c.tangents_in.push_back(h.position + h.in_offset);
		c.tangents_out.push_back(h.position + h.out_offset);
	}
	return c;
}

} // namespace curvenet

#endif
