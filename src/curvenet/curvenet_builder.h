// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
#ifndef CURVENET_CURVENET_BUILDER_H
#define CURVENET_CURVENET_BUILDER_H

// Build a curvenet graph from a flat list of curves (each curve is a
// polyline of knots). ε-merges knot positions across curves so shared
// endpoints become a single graph knot, then classifies each merged
// knot as anchor / regular / intersection per DeGoes22 §3:
//
//   anchor       — incident to exactly 1 spline
//   regular      — incident to 2 splines (either interior of a single
//                  curve, or two open-curve endpoints joining)
//   intersection — incident to 3+ splines (paper's "shared by ≥3
//                  splines" criterion)
//
// Outgoing tangents at each knot are the unit vectors pointing AWAY
// from the knot along each incident segment. They are NOT yet sorted
// CCW around a surface normal — that's the runtime surface-projection
// step's responsibility. Returns them in incidence-list order.

#include <cmath>
#include <cstddef>
#include <vector>

#include "vec3.h"

namespace curvenet {
namespace curvenet_builder {

enum class KnotKind {
	anchor,        // 1 incident segment
	regular,       // 2 incident segments
	intersection,  // ≥3 incident segments (DeGoes22 §3 definition)
};

// One curve's input is a polyline of Vec3 knot positions (in order).
using CurvePoints = std::vector<Vec3>;

// References a knot of a specific input curve.
struct KnotRef {
	int curve_id   = 0;
	int knot_idx   = 0; // index into curves[curve_id]
};

struct CurvenetGraph {
	// Merged knot positions (deduplicated by ε proximity).
	std::vector<Vec3> knot_positions;
	// For each input curve, the merged-knot indices for its knots in
	// order along the curve.
	std::vector<std::vector<int>> curves;
	// For each merged knot, the list of (curve_id, knot_idx) refs that
	// collapsed onto it.
	std::vector<std::vector<KnotRef>> incidence;
};

// Square-distance.
inline double dist_sq(const Vec3 &a, const Vec3 &b) {
	const double dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
	return dx * dx + dy * dy + dz * dz;
}

// Build the curvenet graph: ε-merge knots across curves, record incidence.
inline CurvenetGraph build(const std::vector<CurvePoints> &curves_in, double eps) {
	const double eps_sq = eps * eps;
	CurvenetGraph g;
	g.curves.resize(curves_in.size());

	for (std::size_t c = 0; c < curves_in.size(); ++c) {
		g.curves[c].reserve(curves_in[c].size());
		for (std::size_t k = 0; k < curves_in[c].size(); ++k) {
			const Vec3 p = curves_in[c][k];
			int found = -1;
			for (std::size_t q = 0; q < g.knot_positions.size(); ++q) {
				if (dist_sq(g.knot_positions[q], p) <= eps_sq) {
					found = static_cast<int>(q);
					break;
				}
			}
			if (found < 0) {
				found = static_cast<int>(g.knot_positions.size());
				g.knot_positions.push_back(p);
				g.incidence.push_back({});
			}
			g.curves[c].push_back(found);
			KnotRef ref;
			ref.curve_id = static_cast<int>(c);
			ref.knot_idx = static_cast<int>(k);
			g.incidence[found].push_back(ref);
		}
	}
	return g;
}

// Number of segment endpoints incident to merged knot `k`. Counts each
// incident segment once.
inline int incident_segment_count(const CurvenetGraph &g, std::size_t k) {
	int n = 0;
	for (const KnotRef &r : g.incidence[k]) {
		const std::size_t cn = g.curves[r.curve_id].size();
		if (r.knot_idx > 0) {
			++n; // segment ending at this knot
		}
		if (static_cast<std::size_t>(r.knot_idx) < cn - 1) {
			++n; // segment starting at this knot
		}
	}
	return n;
}

// Classify each merged knot as anchor / regular / intersection.
inline std::vector<KnotKind> classify(const CurvenetGraph &g) {
	std::vector<KnotKind> out(g.knot_positions.size(), KnotKind::regular);
	for (std::size_t k = 0; k < g.knot_positions.size(); ++k) {
		const int n = incident_segment_count(g, k);
		if (n >= 3) {
			out[k] = KnotKind::intersection;
		} else if (n == 1) {
			out[k] = KnotKind::anchor;
		} else {
			out[k] = KnotKind::regular;
		}
	}
	return out;
}

// Unit-length outgoing tangents at each merged knot (one per incident
// segment, in incidence-list order).
inline std::vector<std::vector<Vec3>> outgoing_tangents(const CurvenetGraph &g) {
	std::vector<std::vector<Vec3>> out(g.knot_positions.size());
	for (std::size_t k = 0; k < g.knot_positions.size(); ++k) {
		const Vec3 here = g.knot_positions[k];
		for (const KnotRef &r : g.incidence[k]) {
			const std::vector<int> &indices = g.curves[r.curve_id];
			const std::size_t cn = indices.size();
			if (r.knot_idx > 0) {
				// Tangent toward the previous knot along this curve.
				const Vec3 prev = g.knot_positions[indices[r.knot_idx - 1]];
				const Vec3 d = prev - here;
				const double l = d.length();
				if (l > 0.0) {
					out[k].push_back({ d.x / l, d.y / l, d.z / l });
				}
			}
			if (static_cast<std::size_t>(r.knot_idx) < cn - 1) {
				// Tangent toward the next knot along this curve.
				const Vec3 next_p = g.knot_positions[indices[r.knot_idx + 1]];
				const Vec3 d = next_p - here;
				const double l = d.length();
				if (l > 0.0) {
					out[k].push_back({ d.x / l, d.y / l, d.z / l });
				}
			}
		}
	}
	return out;
}

} // namespace curvenet_builder
} // namespace curvenet

#endif
