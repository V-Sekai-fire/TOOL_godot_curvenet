// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
//
// STUB implementation: returns the input unchanged (every triangle stays a triangle).
// Cycle 7 GREEN step swaps this for LEMON MaxWeightedMatching over the triangle dual graph.

#include "curvenet/tris_to_quads.h"

namespace curvenet {

PolyMesh tris_to_quads(const TriMesh &input, const TrisToQuadsParams &params) {
	(void)params;
	PolyMesh out;
	out.vertices = input.vertices;
	out.faces.reserve(input.triangles.size());
	for (const auto &t : input.triangles) {
		PolyFace f;
		f.count = 3;
		f.v[0] = t[0];
		f.v[1] = t[1];
		f.v[2] = t[2];
		out.faces.push_back(f);
	}
	return out;
}

} // namespace curvenet
