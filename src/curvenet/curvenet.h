// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
#ifndef CURVENET_CURVENET_H
#define CURVENET_CURVENET_H

#include "ngon_patch.h"
#include "vec3.h"

#include <cstdint>
#include <vector>

namespace curvenet {

// One mesh face's boundary curves. For a quad face: 4 boundaries CCW.
// Built up by Cycle 6 from a `tris_to_quads` output by extracting profile-curve
// segments, but during the math layer we accept it directly.
struct BoundFace {
	std::vector<BoundaryCurve> boundaries;
};

// One source-mesh vertex's binding to a face: face index + (s,t) parameter.
// (s,t) is the patch parameter inside `boundaries`.
struct BoundVertex {
	int face_index = -1;
	double s = 0.0;
	double t = 0.0;
};

// Container of per-face boundary curves. Cycle 6 will build this from a
// (mesh, profile_curves) pair; the math layer just consumes it.
struct CurveNet {
	std::vector<BoundFace> faces;
};

// Deform each bound vertex by re-evaluating the patch with the current curves.
// `out` is resized to match `bindings.size()`. Returns out by reference for
// fluent use.
std::vector<Vec3> deform(const CurveNet &net, const std::vector<BoundVertex> &bindings);

} // namespace curvenet

#endif
