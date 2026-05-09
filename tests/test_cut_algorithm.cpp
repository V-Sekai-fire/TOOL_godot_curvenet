// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
//
// Mirrors the Lean cut-algorithm + cut-mesh + cut-mesh-Laplacian
// instance theorems (slices 3, 6, 10, 11, 12, 19, 20).
#include "test_helpers.h"

#include "curvenet/cut_algorithm.h"
#include "curvenet/cut_mesh.h"
#include "curvenet/cut_mesh_laplacian.h"
#include "curvenet/halfedge.h"
#include "curvenet/polygon_laplacian.h"

#include <cstddef>
#include <vector>

using curvenet::Halfedge;
using curvenet::HalfedgeMesh;
using curvenet::OptionalIndex;
using curvenet::Vec3;
using curvenet::is_manifold;
namespace cm  = curvenet::cut_mesh;
namespace ca  = curvenet::cut_algorithm;
namespace cml = curvenet::cut_mesh_laplacian;
namespace pl  = curvenet::polygon_laplacian;

static HalfedgeMesh build_triangle() {
	HalfedgeMesh m;
	m.vertex_count = 3;
	m.face_count = 1;
	m.halfedges = {
		{ 1, OptionalIndex::some(5), 1, OptionalIndex::some(0) },
		{ 2, OptionalIndex::some(4), 2, OptionalIndex::some(0) },
		{ 0, OptionalIndex::some(3), 0, OptionalIndex::some(0) },
		{ 2, OptionalIndex::some(2), 4, OptionalIndex::none()  },
		{ 1, OptionalIndex::some(1), 5, OptionalIndex::none()  },
		{ 0, OptionalIndex::some(0), 3, OptionalIndex::none()  },
	};
	return m;
}

static HalfedgeMesh build_quad() {
	HalfedgeMesh m;
	m.vertex_count = 4;
	m.face_count = 1;
	m.halfedges = {
		{ 1, OptionalIndex::some(7), 1, OptionalIndex::some(0) },
		{ 2, OptionalIndex::some(6), 2, OptionalIndex::some(0) },
		{ 3, OptionalIndex::some(5), 3, OptionalIndex::some(0) },
		{ 0, OptionalIndex::some(4), 0, OptionalIndex::some(0) },
		{ 3, OptionalIndex::some(3), 5, OptionalIndex::none()  },
		{ 2, OptionalIndex::some(2), 6, OptionalIndex::none()  },
		{ 1, OptionalIndex::some(1), 7, OptionalIndex::none()  },
		{ 0, OptionalIndex::some(0), 4, OptionalIndex::none()  },
	};
	return m;
}

static cm::CutMesh build_uncut_quad() {
	cm::CutMesh out;
	out.base = build_quad();
	out.vertex_kind.assign(4, cm::CutVertexKind::mesh_vertex_kind());
	out.segment_of_halfedge.assign(8, -1);
	return out;
}

static int no_sample_col(int, int, bool) { return 0; }

int main() {
	bool ok = true;

	// --- subdivideEdge (slice 10) ---
	ok &= rc::check("subdivide_edge: triangle 4 verts / 8 he / still manifold", [] {
		const HalfedgeMesh out = ca::subdivide_edge(build_triangle(), 0);
		RC_ASSERT(out.vertex_count == 4);
		RC_ASSERT(out.he_count() == 8);
		RC_ASSERT(is_manifold(out));
	});

	// --- splitFace (slice 11) ---
	ok &= rc::check("split_face: quad diagonal -> 4 verts / 10 he / 2 faces / manifold", [] {
		const HalfedgeMesh out = ca::split_face(build_quad(), 0, 2);
		RC_ASSERT(out.vertex_count == 4);
		RC_ASSERT(out.he_count() == 10);
		RC_ASSERT(out.face_count == 2);
		RC_ASSERT(is_manifold(out));
	});

	// --- insertCrack (slice 19) ---
	ok &= rc::check("insert_crack: triangle 4 verts / 8 he / 1 face / manifold", [] {
		const HalfedgeMesh out = ca::insert_crack(build_triangle(), 0);
		RC_ASSERT(out.vertex_count == 4);
		RC_ASSERT(out.he_count() == 8);
		RC_ASSERT(out.face_count == 1);
		RC_ASSERT(is_manifold(out));
	});

	// --- CutMesh wrappers (slice 12) ---
	ok &= rc::check("cut_mesh: uncut quad partition of unity holds", [] {
		const cm::CutMesh m = build_uncut_quad();
		RC_ASSERT(cm::partition_of_unity(m, no_sample_col));
		RC_ASSERT(cm::vt_c_is_zero(m, no_sample_col));
	});

	ok &= rc::check("subdivide_edge_cm: partition still holds", [] {
		const cm::CutMesh m = build_uncut_quad();
		const cm::CutMesh out = ca::subdivide_edge_cm(m, 0,
			cm::CutVertexKind::sample_kind(0, 0, false));
		RC_ASSERT(out.vertex_count() == 5);
		RC_ASSERT(cm::partition_of_unity(out, no_sample_col));
		RC_ASSERT(cm::vt_c_is_zero(out, no_sample_col));
	});

	ok &= rc::check("split_face_cm: new halfedges tagged with seg id", [] {
		const cm::CutMesh m = build_uncut_quad();
		const cm::CutMesh out = ca::split_face_cm(m, 0, 2, 7);
		RC_ASSERT(out.he_count() == 10);
		RC_ASSERT(out.segment_of_halfedge[8] == 7);
		RC_ASSERT(out.segment_of_halfedge[9] == 7);
	});

	// --- CutMeshLaplacian (slice 6) ---
	ok &= rc::check("VtLhV uncut equilateral triangle: row sums vanish", [] {
		const std::vector<Vec3> positions = {
			{ 0.0, 0.0, 0.0 },
			{ 1.0, 0.0, 0.0 },
			{ 0.5, 0.0, 0.8660254037844386 } };
		cm::CutMesh m;
		m.base = build_triangle();
		m.vertex_kind.assign(3, cm::CutVertexKind::mesh_vertex_kind());
		m.segment_of_halfedge.assign(6, -1);
		const std::vector<double> M = cml::assemble_vt_lh_v(m, positions);
		RC_ASSERT(pl::row_sums_within(M, 3, 1e-12));
		RC_ASSERT(pl::is_symmetric_within(M, 3, 1e-12));
	});

	// --- Slice 20: chain of length 1 across two-tri strip ---
	auto build_two_tri_strip = []() -> HalfedgeMesh {
		HalfedgeMesh m;
		m.vertex_count = 4;
		m.face_count = 2;
		m.halfedges = {
			{ 1, OptionalIndex::some(7), 1, OptionalIndex::some(0) }, // 0
			{ 2, OptionalIndex::some(3), 2, OptionalIndex::some(0) }, // 1
			{ 0, OptionalIndex::some(6), 0, OptionalIndex::some(0) }, // 2
			{ 1, OptionalIndex::some(1), 4, OptionalIndex::some(1) }, // 3
			{ 3, OptionalIndex::some(9), 5, OptionalIndex::some(1) }, // 4
			{ 2, OptionalIndex::some(8), 3, OptionalIndex::some(1) }, // 5
			{ 2, OptionalIndex::some(2), 8, OptionalIndex::none()  }, // 6
			{ 0, OptionalIndex::some(0), 6, OptionalIndex::none()  }, // 7
			{ 3, OptionalIndex::some(5), 9, OptionalIndex::none()  }, // 8
			{ 1, OptionalIndex::some(4), 7, OptionalIndex::none()  }, // 9
		};
		return m;
	};

	ok &= rc::check("chain length 1: 5 verts / 16 he / 4 faces / manifold", [build_two_tri_strip] {
		const HalfedgeMesh m1 = ca::subdivide_edge(build_two_tri_strip(), 1);
		const HalfedgeMesh m2 = ca::split_face(m1, 2, 1);
		const HalfedgeMesh m3 = ca::split_face(m2, 3, 4);
		RC_ASSERT(m3.vertex_count == 5);
		RC_ASSERT(m3.he_count() == 16);
		RC_ASSERT(m3.face_count == 4);
		RC_ASSERT(is_manifold(m3));
	});

	return ok ? 0 : 1;
}
