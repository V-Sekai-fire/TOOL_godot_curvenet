// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
//
// Verifies `halfedge_builder::from_triangles` produces manifold
// halfedge meshes for the canonical examples (single triangle, two-tri
// strip, closed tetrahedron).
#include "test_helpers.h"

#include "curvenet/halfedge.h"
#include "curvenet/halfedge_builder.h"

#include <cstddef>
#include <vector>

using curvenet::HalfedgeMesh;
using curvenet::is_manifold;
namespace hb = curvenet::halfedge_builder;

int main() {
	bool ok = true;

	ok &= rc::check("single triangle: 3 verts / 1 face / 6 halfedges / manifold", [] {
		const std::vector<int> tris = { 0, 1, 2 };
		const HalfedgeMesh m = hb::from_triangles(3, tris);
		RC_ASSERT(m.vertex_count == 3);
		RC_ASSERT(m.face_count   == 1);
		RC_ASSERT(m.he_count()   == 6);
		RC_ASSERT(is_manifold(m));
	});

	ok &= rc::check("two-tri strip (shared edge 1-2): 4 verts / 2 face / 10 he / manifold", [] {
		const std::vector<int> tris = {
			0, 1, 2,        // face 0
			2, 1, 3 };      // face 1, sharing edge 1-2 (note CCW orientation)
		const HalfedgeMesh m = hb::from_triangles(4, tris);
		RC_ASSERT(m.vertex_count == 4);
		RC_ASSERT(m.face_count   == 2);
		RC_ASSERT(m.he_count()   == 10);
		RC_ASSERT(is_manifold(m));
	});

	ok &= rc::check("closed tetrahedron: 4 verts / 4 faces / 12 halfedges / manifold", [] {
		// Tetra: 4 vertices, 4 triangle faces, no boundary.
		const std::vector<int> tris = {
			0, 2, 1,        // face 0
			0, 1, 3,        // face 1
			0, 3, 2,        // face 2
			1, 2, 3 };      // face 3
		const HalfedgeMesh m = hb::from_triangles(4, tris);
		RC_ASSERT(m.vertex_count == 4);
		RC_ASSERT(m.face_count   == 4);
		RC_ASSERT(m.he_count()   == 12); // closed: 6 edges × 2 halfedges, no boundaries
		RC_ASSERT(is_manifold(m));
	});

	// Euler V-E+F = 2 for both genus-0 examples.
	ok &= rc::check("single triangle Euler V-E+F = 2", [] {
		const std::vector<int> tris = { 0, 1, 2 };
		const HalfedgeMesh m = hb::from_triangles(3, tris);
		const std::size_t V = m.vertex_count;
		const std::size_t F = m.face_count + 1; // + outer
		const std::size_t E = m.he_count() / 2;
		RC_ASSERT(V + F - E == 2);
	});

	ok &= rc::check("closed tetrahedron Euler V-E+F = 2", [] {
		const std::vector<int> tris = {
			0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3 };
		const HalfedgeMesh m = hb::from_triangles(4, tris);
		const std::size_t V = m.vertex_count;
		const std::size_t F = m.face_count;        // closed mesh, no outer
		const std::size_t E = m.he_count() / 2;
		RC_ASSERT(V + F - E == 2);
	});

	return ok ? 0 : 1;
}
