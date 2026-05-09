// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
#ifndef CURVENET_SURFACE_PROJECTION_H
#define CURVENET_SURFACE_PROJECTION_H

// Project curvenet knots onto a triangle mesh's surface and return the
// projection result categorised per DeGoes22 §4.1 ("vertex / edge /
// face-sample"). This first slice supports VERTEX projection only —
// each knot maps to its closest mesh vertex. Edge and face projections
// (which require running slices 10–11/20 on the cut-mesh) come later.
//
// Building a cut-mesh from a vertex-projected knot list reduces to a
// pure annotation step: tag the projected mesh vertex with the
// `sample` kind (curve_id / sample_idx / side) — no halfedge surgery
// required. That's `promote_vertex_samples` here.

#include <cstddef>
#include <limits>
#include <vector>

#include "cut_mesh.h"
#include "curvenet_builder.h"
#include "vec3.h"

namespace curvenet {
namespace surface_projection {

enum class ProjectionKind {
	vertex,             // closest point on mesh is a vertex
	edge_intersection,  // closest is an edge interior   (TODO: slice 10/12)
	face_interior,      // closest is a face interior    (TODO: slice 11/20)
};

struct ProjectedKnot {
	ProjectionKind kind   = ProjectionKind::vertex;
	int            mesh_index = -1;   // vertex_idx (kind=vertex), or halfedge / face for the other cases
	Vec3           position;
};

// Project each input knot to its closest mesh vertex. Returns one
// ProjectedKnot per input knot, all of kind `vertex`.
inline std::vector<ProjectedKnot>
project_to_vertices(const std::vector<Vec3> &knots,
                     const std::vector<Vec3> &mesh_positions) {
	std::vector<ProjectedKnot> out;
	out.reserve(knots.size());
	for (const Vec3 &k : knots) {
		int best_idx = -1;
		double best_d = std::numeric_limits<double>::infinity();
		for (std::size_t v = 0; v < mesh_positions.size(); ++v) {
			const double dx = mesh_positions[v].x - k.x;
			const double dy = mesh_positions[v].y - k.y;
			const double dz = mesh_positions[v].z - k.z;
			const double d  = dx * dx + dy * dy + dz * dz;
			if (d < best_d) {
				best_d   = d;
				best_idx = static_cast<int>(v);
			}
		}
		ProjectedKnot pk;
		pk.kind       = ProjectionKind::vertex;
		pk.mesh_index = best_idx;
		pk.position   = (best_idx >= 0) ? mesh_positions[best_idx] : k;
		out.push_back(pk);
	}
	return out;
}

// Promote each vertex-projected knot to a `sample` kind in the supplied
// CutMesh. Returns the mapping `knot_to_sample_col`: per input knot,
// the C-column index that the deformer should use for that knot.
// Multiple knots projecting to the same vertex collapse to the same
// column (the first one encountered).
//
// Pre: every entry in `projections` has kind == vertex.
inline std::vector<int>
promote_vertex_samples(cut_mesh::CutMesh &cm,
                        const std::vector<ProjectedKnot> &projections) {
	std::vector<int> knot_to_col(projections.size(), -1);
	int next_col = 0;
	// First pass: assign columns. Two projections sharing the same
	// mesh vertex share a column.
	std::vector<int> vertex_to_col(cm.vertex_count(), -1);
	for (std::size_t i = 0; i < projections.size(); ++i) {
		if (projections[i].kind != ProjectionKind::vertex) {
			continue; // skip non-vertex (TODO future slices)
		}
		const int v = projections[i].mesh_index;
		if (v < 0) {
			continue;
		}
		if (vertex_to_col[v] < 0) {
			vertex_to_col[v] = next_col++;
		}
		knot_to_col[i] = vertex_to_col[v];
	}
	// Second pass: tag the cut-mesh vertex_kind with sample metadata.
	// We use curve_id = column, sample_idx = 0, side = false; the
	// deformer's sample-column packer just maps (curve_id, _, _) -> col.
	for (std::size_t v = 0; v < cm.vertex_count(); ++v) {
		const int col = vertex_to_col[v];
		if (col >= 0) {
			cm.vertex_kind[v] = cut_mesh::CutVertexKind::sample_kind(col, 0, false);
		}
	}
	return knot_to_col;
}

} // namespace surface_projection
} // namespace curvenet

#endif
