// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
//
// Verifies `surface_projection::project_to_vertices` and
// `promote_vertex_samples` correctly map curvenet knots onto a small
// triangle mesh and tag the cut-mesh vertex kinds.
#include "test_helpers.h"

#include "curvenet/cut_mesh.h"
#include "curvenet/halfedge.h"
#include "curvenet/halfedge_builder.h"
#include "curvenet/surface_projection.h"

#include <cstddef>
#include <vector>

using curvenet::HalfedgeMesh;
using curvenet::Vec3;
namespace cm = curvenet::cut_mesh;
namespace hb = curvenet::halfedge_builder;
namespace sp = curvenet::surface_projection;

int main() {
	bool ok = true;

	// 4-vertex unit square, two triangles, used for projection tests.
	const std::vector<Vec3> mesh_positions = {
		{ 0.0, 0.0, 0.0 },
		{ 1.0, 0.0, 0.0 },
		{ 1.0, 0.0, 1.0 },
		{ 0.0, 0.0, 1.0 } };
	const std::vector<int> tris = { 0, 1, 2, 0, 2, 3 };

	ok &= rc::check("project_to_vertices: a knot at (1.0, 0.4, 0.0) -> vertex 1", [mesh_positions] {
		const std::vector<Vec3> knots = { { 1.0, 0.4, 0.0 } };
		const auto pj = sp::project_to_vertices(knots, mesh_positions);
		RC_ASSERT(pj.size() == 1);
		RC_ASSERT(pj[0].kind == sp::ProjectionKind::vertex);
		RC_ASSERT(pj[0].mesh_index == 1);
	});

	ok &= rc::check("project_to_vertices: 4 corner knots map to 4 corner verts", [mesh_positions] {
		const std::vector<Vec3> knots = {
			{  0.0, 0.4, 0.0 },
			{  1.0, 0.4, 0.0 },
			{  1.0, 0.4, 1.0 },
			{  0.0, 0.4, 1.0 } };
		const auto pj = sp::project_to_vertices(knots, mesh_positions);
		for (std::size_t i = 0; i < pj.size(); ++i) {
			RC_ASSERT(pj[i].mesh_index == static_cast<int>(i));
		}
	});

	ok &= rc::check("promote_vertex_samples: tags the projected vertices", [mesh_positions, tris] {
		cm::CutMesh m;
		m.base = hb::from_triangles(4, tris);
		m.vertex_kind.assign(4, cm::CutVertexKind::mesh_vertex_kind());
		m.segment_of_halfedge.assign(m.base.he_count(), -1);

		const std::vector<Vec3> knots = {
			{ 0.0, 0.4, 0.0 }, { 1.0, 0.4, 0.0 } };
		const auto pj = sp::project_to_vertices(knots, mesh_positions);
		const std::vector<int> knot_to_col = sp::promote_vertex_samples(m, pj);

		// Vertex 0 → column 0; vertex 1 → column 1.
		RC_ASSERT(m.vertex_kind[0].tag == cm::CutVertexKindTag::sample);
		RC_ASSERT(m.vertex_kind[1].tag == cm::CutVertexKindTag::sample);
		RC_ASSERT(m.vertex_kind[2].tag == cm::CutVertexKindTag::mesh_vertex);
		RC_ASSERT(m.vertex_kind[3].tag == cm::CutVertexKindTag::mesh_vertex);

		// knot_to_col returns the column assignments.
		RC_ASSERT(knot_to_col[0] == 0);
		RC_ASSERT(knot_to_col[1] == 1);
	});

	ok &= rc::check("promote_vertex_samples: two knots on same vertex share column", [mesh_positions, tris] {
		cm::CutMesh m;
		m.base = hb::from_triangles(4, tris);
		m.vertex_kind.assign(4, cm::CutVertexKind::mesh_vertex_kind());
		m.segment_of_halfedge.assign(m.base.he_count(), -1);

		// Two knots both project to vertex 0 (the (0, 0, 0) corner).
		const std::vector<Vec3> knots = {
			{ 0.0, 0.4, 0.1 }, { 0.0, 0.4, 0.0 } };
		const auto pj = sp::project_to_vertices(knots, mesh_positions);
		const std::vector<int> knot_to_col = sp::promote_vertex_samples(m, pj);
		// Both knots resolve to column 0.
		RC_ASSERT(knot_to_col[0] == 0);
		RC_ASSERT(knot_to_col[1] == 0);
		// Vertex 0 promoted, vertex 1/2/3 still mesh-vertex.
		RC_ASSERT(m.vertex_kind[0].tag == cm::CutVertexKindTag::sample);
	});

	return ok ? 0 : 1;
}
