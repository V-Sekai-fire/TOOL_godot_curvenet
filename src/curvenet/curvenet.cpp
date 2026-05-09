// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
#include "curvenet/curvenet.h"

namespace curvenet {

std::vector<Vec3> deform(const CurveNet &net, const std::vector<BoundVertex> &bindings) {
	std::vector<Vec3> out;
	out.reserve(bindings.size());
	for (const auto &b : bindings) {
		if (b.face_index < 0 || b.face_index >= static_cast<int>(net.faces.size())) {
			out.push_back(Vec3{ 0.0, 0.0, 0.0 });
			continue;
		}
		NgonPatch p;
		p.boundaries = net.faces[b.face_index].boundaries;
		out.push_back(p.evaluate(b.s, b.t));
	}
	return out;
}

} // namespace curvenet
