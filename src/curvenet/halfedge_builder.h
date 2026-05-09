// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
#ifndef CURVENET_HALFEDGE_BUILDER_H
#define CURVENET_HALFEDGE_BUILDER_H

// Build a HalfedgeMesh from a flat triangle index list. Used to bridge
// runtime triangle-soup input (e.g. Godot's ArrayMesh ARRAY_INDEX
// PackedInt32Array) into the cut-mesh data structure.
//
// Conventions: triangle f has vertices (tri[3f], tri[3f+1], tri[3f+2])
// in CCW order. Output halfedge indexing:
//
//   he 3f, 3f+1, 3f+2  — interior, in CCW face-loop order
//   he >= 3·n_tris     — boundary halfedges added for edges with no
//                        opposite interior counterpart
//
// Boundary halfedges are linked into outer-face loops by walking the
// "next boundary at this target vertex" relationship.

#include <cstddef>
#include <map>
#include <utility>
#include <vector>

#include "halfedge.h"

namespace curvenet {
namespace halfedge_builder {

// Build a HalfedgeMesh from `vertex_count` and a flat list of triangle
// indices (3 ints per triangle).
inline HalfedgeMesh from_triangles(std::size_t vertex_count,
                                     const std::vector<int> &tri_indices) {
	HalfedgeMesh m;
	m.vertex_count = vertex_count;
	const std::size_t n_tris = tri_indices.size() / 3;
	m.face_count = n_tris;
	m.halfedges.reserve(3 * n_tris * 2); // worst case all-boundary

	// Step 1: interior halfedges. For triangle f with verts (t0, t1, t2):
	//   he 3f:   t0 -> t1, next = 3f+1
	//   he 3f+1: t1 -> t2, next = 3f+2
	//   he 3f+2: t2 -> t0, next = 3f
	for (std::size_t f = 0; f < n_tris; ++f) {
		const int t0 = tri_indices[3 * f + 0];
		const int t1 = tri_indices[3 * f + 1];
		const int t2 = tri_indices[3 * f + 2];
		const int base = static_cast<int>(3 * f);
		const OptionalIndex face = OptionalIndex::some(static_cast<int>(f));
		m.halfedges.push_back({ t1, OptionalIndex::none(), base + 1, face });
		m.halfedges.push_back({ t2, OptionalIndex::none(), base + 2, face });
		m.halfedges.push_back({ t0, OptionalIndex::none(), base,     face });
	}

	// Source vertex for interior halfedge h. For he 3f+local in triangle
	// f, source equals tri_indices[3f + local].
	auto source_of_interior = [&](int h) -> int {
		const int f = h / 3;
		const int local = h % 3;
		return tri_indices[3 * f + local];
	};

	// Step 2: pair up interior twins by (source, target) edge.
	std::map<std::pair<int, int>, int> dir_map;
	const int n_interior = static_cast<int>(m.halfedges.size());
	for (int i = 0; i < n_interior; ++i) {
		dir_map[{ source_of_interior(i), m.halfedges[i].target }] = i;
	}

	std::vector<int> need_boundary;
	for (int i = 0; i < n_interior; ++i) {
		auto it = dir_map.find({ m.halfedges[i].target, source_of_interior(i) });
		if (it != dir_map.end()) {
			m.halfedges[i].twin = OptionalIndex::some(it->second);
		} else {
			need_boundary.push_back(i);
		}
	}

	// Step 3: each interior halfedge without an interior twin gets a
	// paired boundary halfedge in the opposite direction.
	const std::size_t boundary_start = m.halfedges.size();
	for (const int i : need_boundary) {
		const int bh_target = source_of_interior(i);
		m.halfedges.push_back({
			bh_target,
			OptionalIndex::some(i),
			0,                          // placeholder, fixed in step 4
			OptionalIndex::none()
		});
		m.halfedges[i].twin =
			OptionalIndex::some(static_cast<int>(m.halfedges.size() - 1));
	}

	// Step 4: link boundary halfedges into outer-face loops.
	//
	// For boundary halfedge bh:
	//   source(bh) = target of its interior twin (since twins go opposite
	//                directions across the same edge).
	//   bh.next = the boundary halfedge whose source equals bh.target.
	std::map<int, int> boundary_at_source;
	for (std::size_t i = boundary_start; i < m.halfedges.size(); ++i) {
		const int twin_idx = m.halfedges[i].twin.unwrap();
		const int my_source = m.halfedges[twin_idx].target;
		boundary_at_source[my_source] = static_cast<int>(i);
	}
	for (std::size_t i = boundary_start; i < m.halfedges.size(); ++i) {
		const int my_target = m.halfedges[i].target;
		auto it = boundary_at_source.find(my_target);
		if (it != boundary_at_source.end()) {
			m.halfedges[i].next = it->second;
		}
	}

	return m;
}

} // namespace halfedge_builder
} // namespace curvenet

#endif
