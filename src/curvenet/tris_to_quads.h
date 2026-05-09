// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
#ifndef CURVENET_TRIS_TO_QUADS_H
#define CURVENET_TRIS_TO_QUADS_H

#include "vec3.h"

#include <array>
#include <cstdint>
#include <vector>

namespace curvenet {

struct TriMesh {
	std::vector<Vec3> vertices;
	std::vector<std::array<int, 3>> triangles; // CCW indices into `vertices`
};

// A face is either a triangle (3 indices, last entry = -1 ignored) or a quad (4 indices).
struct PolyFace {
	int count = 0; // 3 or 4
	int v[4] = { -1, -1, -1, -1 };
};

struct PolyMesh {
	std::vector<Vec3> vertices;
	std::vector<PolyFace> faces;
};

// Edge length is the only quality input the original Rulesobeyer addon uses.
// Weight = 1 + 0.1 * length / max_length, so longer dissolved edges break ties.
struct TrisToQuadsParams {
	double length_tiebreak = 0.1;
};

// STUB.
PolyMesh tris_to_quads(const TriMesh &input, const TrisToQuadsParams &params = {});

} // namespace curvenet

#endif
