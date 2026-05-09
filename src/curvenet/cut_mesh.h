// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
#ifndef CURVENET_CUT_MESH_H
#define CURVENET_CUT_MESH_H

// C++ mirror of `lean/Curvenet/CutMesh.lean`. The CutMesh data layer:
// halfedge mesh + per-vertex kind tag + per-halfedge segment annotation,
// plus the V/C sparse mapping helpers and partition-of-unity invariant.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "halfedge.h"

namespace curvenet {
namespace cut_mesh {

enum class CutVertexKindTag : std::uint8_t {
	mesh_vertex,
	sample,
	edge_intersection,
};

struct CutVertexKind {
	CutVertexKindTag tag = CutVertexKindTag::mesh_vertex;
	int curve_id   = 0;
	int sample_idx = 0;
	bool side       = false;
	int mesh_edge  = 0;

	static CutVertexKind mesh_vertex_kind() {
		return CutVertexKind{ CutVertexKindTag::mesh_vertex, 0, 0, false, 0 };
	}
	static CutVertexKind sample_kind(int curve, int idx, bool which_side) {
		return CutVertexKind{ CutVertexKindTag::sample, curve, idx, which_side, 0 };
	}
	static CutVertexKind edge_intersection_kind(int mesh_edge_idx) {
		return CutVertexKind{ CutVertexKindTag::edge_intersection, 0, 0, false, mesh_edge_idx };
	}
};

struct CutMesh {
	HalfedgeMesh                base;
	std::vector<CutVertexKind>  vertex_kind;
	// Indexed by halfedge index; -1 sentinel = none, else the segment id.
	std::vector<int>            segment_of_halfedge;

	std::size_t he_count()      const { return base.he_count(); }
	std::size_t vertex_count()  const { return base.vertex_count; }
};

// V column for halfedge h: target(h) if target is a mesh-vertex, else
// none (-1). Halfedges whose target is a sample contribute to C, not V.
inline int v_column_of(const CutMesh &m, std::size_t h) {
	const Halfedge &he = m.base.halfedges[h];
	const CutVertexKind &k = m.vertex_kind[he.target];
	if (k.tag == CutVertexKindTag::mesh_vertex) {
		return he.target;
	}
	return -1;
}

// C column for halfedge h, packed via the supplied function. Returns -1
// when the halfedge target is not a sample.
template <typename SampleColumnFn>
inline int c_column_of(const CutMesh &m, std::size_t h, SampleColumnFn &&pack) {
	const Halfedge &he = m.base.halfedges[h];
	const CutVertexKind &k = m.vertex_kind[he.target];
	if (k.tag == CutVertexKindTag::sample) {
		return pack(k.curve_id, k.sample_idx, k.side);
	}
	return -1;
}

// Every halfedge contributes to exactly one of V or C, never both,
// never neither.
template <typename SampleColumnFn>
inline bool partition_of_unity(const CutMesh &m, SampleColumnFn &&pack) {
	const std::size_t n = m.he_count();
	for (std::size_t h = 0; h < n; ++h) {
		const int v = v_column_of(m, h);
		const int c = c_column_of(m, h, pack);
		const bool has_v = (v >= 0);
		const bool has_c = (c >= 0);
		if (has_v == has_c) {
			return false; // either both or neither
		}
	}
	return true;
}

template <typename SampleColumnFn>
inline bool vt_c_is_zero(const CutMesh &m, SampleColumnFn &&pack) {
	const std::size_t n = m.he_count();
	for (std::size_t h = 0; h < n; ++h) {
		if (v_column_of(m, h) >= 0 && c_column_of(m, h, pack) >= 0) {
			return false;
		}
	}
	return true;
}

} // namespace cut_mesh
} // namespace curvenet

#endif
