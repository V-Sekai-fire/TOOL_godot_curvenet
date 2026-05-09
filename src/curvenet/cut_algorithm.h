// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
#ifndef CURVENET_CUT_ALGORITHM_H
#define CURVENET_CUT_ALGORITHM_H

// C++ mirror of `lean/Curvenet/CutAlgorithm.lean`. Cut-mesh surgery
// primitives: edge subdivision, face split, dead-end crack insertion,
// and CutMesh wrappers that maintain per-vertex kind + per-halfedge
// segment annotations alongside the underlying halfedge edits.

#include <cstddef>
#include <vector>

#include "cut_mesh.h"
#include "halfedge.h"

namespace curvenet {
namespace cut_algorithm {

// Subdivide the edge containing halfedge `h`, inserting a new vertex
// in the middle. Adds +1 vertex, +2 halfedges. Manifold invariants
// preserved.
inline HalfedgeMesh subdivide_edge(const HalfedgeMesh &m, std::size_t h) {
	const std::size_t nh = m.he_count();
	const std::size_t nv = m.vertex_count;
	const Halfedge he_orig = m.halfedges[h];
	const std::size_t twin_idx = he_orig.twin.has_value()
		? static_cast<std::size_t>(he_orig.twin.unwrap())
		: h;
	const Halfedge he_twin = m.halfedges[twin_idx];

	const std::size_t c_idx        = nv;
	const std::size_t h_new_idx    = nh;
	const std::size_t twin_new_idx = nh + 1;

	HalfedgeMesh out = m;
	// h: src → c
	out.halfedges[h] = {
		static_cast<int>(c_idx),
		OptionalIndex::some(static_cast<int>(twin_new_idx)),
		static_cast<int>(h_new_idx),
		he_orig.face
	};
	// twin: src(twin) → c
	out.halfedges[twin_idx] = {
		static_cast<int>(c_idx),
		OptionalIndex::some(static_cast<int>(h_new_idx)),
		static_cast<int>(twin_new_idx),
		he_twin.face
	};
	// h_new: c → old target of h
	out.halfedges.push_back({
		he_orig.target,
		OptionalIndex::some(static_cast<int>(twin_idx)),
		he_orig.next,
		he_orig.face
	});
	// twin_new: c → old target of twin (i.e. src of h)
	out.halfedges.push_back({
		he_twin.target,
		OptionalIndex::some(static_cast<int>(h)),
		he_twin.next,
		he_twin.face
	});
	out.vertex_count = nv + 1;
	return out;
}

// Split a face by connecting target(a) to target(b) with a new segment.
// Adds +0 vertices, +2 halfedges, +1 face.
inline HalfedgeMesh split_face(const HalfedgeMesh &m, std::size_t a, std::size_t b) {
	const std::size_t nh = m.he_count();
	const std::size_t nf = m.face_count;
	const Halfedge he_a = m.halfedges[a];
	const Halfedge he_b = m.halfedges[b];
	const int old_next_a = he_a.next;
	const int old_next_b = he_b.next;
	const std::size_t f_prime = nf;
	const std::size_t h_ab    = nh;
	const std::size_t h_ba    = nh + 1;

	HalfedgeMesh out = m;
	// Re-tag halfedges from old_next_b up to (but not including) a.
	std::size_t cur = static_cast<std::size_t>(old_next_b);
	std::size_t steps = 0;
	while (cur != a && steps < nh) {
		out.halfedges[cur].face = OptionalIndex::some(static_cast<int>(f_prime));
		cur = out.halfedges[cur].next;
		++steps;
	}
	out.halfedges[a].face = OptionalIndex::some(static_cast<int>(f_prime));
	out.halfedges[a].next = static_cast<int>(h_ab);
	out.halfedges[b].next = static_cast<int>(h_ba);
	out.halfedges.push_back({
		he_b.target,
		OptionalIndex::some(static_cast<int>(h_ba)),
		old_next_b,
		OptionalIndex::some(static_cast<int>(f_prime))
	});
	out.halfedges.push_back({
		he_a.target,
		OptionalIndex::some(static_cast<int>(h_ab)),
		old_next_a,
		he_a.face
	});
	out.face_count = nf + 1;
	return out;
}

// Insert a dead-end crack (slit) inside a face anchored at the boundary
// vertex `target(h_anchor)`. Adds +1 vertex, +2 halfedges, +0 faces.
// The standard `is_manifold` invariants are preserved.
inline HalfedgeMesh insert_crack(const HalfedgeMesh &m, std::size_t h_anchor) {
	const std::size_t nh = m.he_count();
	const std::size_t nv = m.vertex_count;
	const Halfedge he_anchor = m.halfedges[h_anchor];
	const int v_boundary = he_anchor.target;
	const int old_next = he_anchor.next;
	const std::size_t c_idx = nv;
	const std::size_t h_out = nh;
	const std::size_t h_in  = nh + 1;

	HalfedgeMesh out = m;
	out.halfedges[h_anchor].next = static_cast<int>(h_out);
	out.halfedges.push_back({
		static_cast<int>(c_idx),
		OptionalIndex::some(static_cast<int>(h_in)),
		static_cast<int>(h_in),
		he_anchor.face
	});
	out.halfedges.push_back({
		v_boundary,
		OptionalIndex::some(static_cast<int>(h_out)),
		old_next,
		he_anchor.face
	});
	out.vertex_count = nv + 1;
	return out;
}

// CutMesh wrappers maintain per-vertex kind + per-halfedge segment
// annotations alongside the halfedge surgery.

inline cut_mesh::CutMesh subdivide_edge_cm(const cut_mesh::CutMesh &cm,
                                              std::size_t h,
                                              const cut_mesh::CutVertexKind &new_kind) {
	cut_mesh::CutMesh out;
	out.base = subdivide_edge(cm.base, h);
	out.vertex_kind = cm.vertex_kind;
	out.vertex_kind.push_back(new_kind);
	out.segment_of_halfedge = cm.segment_of_halfedge;
	const int seg_of_h = cm.segment_of_halfedge[h];
	out.segment_of_halfedge.push_back(seg_of_h);
	out.segment_of_halfedge.push_back(seg_of_h);
	return out;
}

inline cut_mesh::CutMesh split_face_cm(const cut_mesh::CutMesh &cm,
                                          std::size_t a, std::size_t b,
                                          int seg_id) {
	cut_mesh::CutMesh out;
	out.base = split_face(cm.base, a, b);
	out.vertex_kind = cm.vertex_kind;
	out.segment_of_halfedge = cm.segment_of_halfedge;
	out.segment_of_halfedge.push_back(seg_id);
	out.segment_of_halfedge.push_back(seg_id);
	return out;
}

inline cut_mesh::CutMesh insert_crack_cm(const cut_mesh::CutMesh &cm,
                                            std::size_t h_anchor,
                                            int seg_id,
                                            const cut_mesh::CutVertexKind &new_kind) {
	cut_mesh::CutMesh out;
	out.base = insert_crack(cm.base, h_anchor);
	out.vertex_kind = cm.vertex_kind;
	out.vertex_kind.push_back(new_kind);
	out.segment_of_halfedge = cm.segment_of_halfedge;
	out.segment_of_halfedge.push_back(seg_id);
	out.segment_of_halfedge.push_back(seg_id);
	return out;
}

} // namespace cut_algorithm
} // namespace curvenet

#endif
