// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
#ifndef CURVENET_BINDING_H
#define CURVENET_BINDING_H

#include "bilinear_inverse.h"
#include "curvenet.h"
#include "tris_to_quads.h"
#include "vec3.h"

#include <cstdint>
#include <limits>
#include <vector>

namespace curvenet {

// Bind every vertex of a PolyMesh to its closest face's bilinear domain.
// For each vertex V:
//   - For each quad face (count==4), solve the bilinear inverse to get (s,t)
//     and the residual (distance from V to the bilinear surface).
//   - Pick the face with the smallest residual; that's the owning face.
//   - Triangle faces are skipped (cycle 4 N=3 path will land later); vertices
//     whose closest face is a triangle bind to face_index = -1 with (s,t)=(0,0).
//
// `face_corners` extracts (P00, P10, P11, P01) from a PolyFace's CCW indices.
// Returns a vector parallel to mesh.vertices.
inline std::vector<BoundVertex> bind_polymesh(const PolyMesh &mesh) {
	std::vector<BoundVertex> out(mesh.vertices.size());
	for (std::size_t v_idx = 0; v_idx < mesh.vertices.size(); ++v_idx) {
		const Vec3 &V = mesh.vertices[v_idx];
		double best = std::numeric_limits<double>::infinity();
		int best_face = -1;
		double best_s = 0.0;
		double best_t = 0.0;
		for (std::size_t f_idx = 0; f_idx < mesh.faces.size(); ++f_idx) {
			const PolyFace &f = mesh.faces[f_idx];
			if (f.count != 4) {
				continue;
			}
			const Vec3 &P00 = mesh.vertices[f.v[0]];
			const Vec3 &P10 = mesh.vertices[f.v[1]];
			const Vec3 &P11 = mesh.vertices[f.v[2]];
			const Vec3 &P01 = mesh.vertices[f.v[3]];
			BilinearInverseResult r = solve_bilinear_inverse(V, P00, P10, P11, P01);
			if (r.residual < best) {
				best = r.residual;
				best_face = static_cast<int>(f_idx);
				best_s = r.s;
				best_t = r.t;
			}
		}
		BoundVertex bv;
		bv.face_index = best_face;
		bv.s = best_s;
		bv.t = best_t;
		out[v_idx] = bv;
	}
	return out;
}

} // namespace curvenet

#endif
