// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
#ifndef CURVENET_SEGMENT_GRADIENT_H
#define CURVENET_SEGMENT_GRADIENT_H

// C++ mirror of `lean/Curvenet/SegmentGradient.lean`. Per-segment
// deformation-gradient dispatcher. Picks between the slice-4 isolated
// formula and the slice-15 intersection formula based on segment kind.

#include <utility>

#include "scaled_frames.h"
#include "vec3.h"

namespace curvenet {
namespace segment_gradient {

inline scaled_frames::Mat3 isolated(const Vec3 &rest_start, const Vec3 &rest_end,
                                     const Vec3 &posed_start, const Vec3 &posed_end) {
	return scaled_frames::isolated_segment_gradient(
		rest_start, rest_end, posed_start, posed_end);
}

inline std::pair<scaled_frames::Mat3, scaled_frames::Mat3>
intersection_pair(const scaled_frames::Mat3 &rest_plus,
                   const scaled_frames::Mat3 &posed_plus,
                   const scaled_frames::Mat3 &rest_minus,
                   const scaled_frames::Mat3 &posed_minus) {
	return { scaled_frames::deformation_gradient(rest_plus,  posed_plus),
	         scaled_frames::deformation_gradient(rest_minus, posed_minus) };
}

} // namespace segment_gradient
} // namespace curvenet

#endif
