// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
//
// Mirrors `lean/Curvenet/Halfedge.lean`'s instance theorems for the
// triangle and quad examples. Hand-built halfedge meshes are checked
// against `is_manifold`.
#include "test_helpers.h"

#include "curvenet/halfedge.h"

using curvenet::Halfedge;
using curvenet::HalfedgeMesh;
using curvenet::OptionalIndex;
using curvenet::is_manifold;

// Build the slice-1 triangle: 3 verts, 1 interior face (id 0), 6 halfedges
// (3 interior + 3 boundary).
static HalfedgeMesh build_triangle() {
	HalfedgeMesh m;
	m.vertex_count = 3;
	m.face_count = 1;
	m.halfedges = {
		// Interior CCW: 0->1, 1->2, 2->0.
		{ 1, OptionalIndex::some(5), 1, OptionalIndex::some(0) }, // he 0
		{ 2, OptionalIndex::some(4), 2, OptionalIndex::some(0) }, // he 1
		{ 0, OptionalIndex::some(3), 0, OptionalIndex::some(0) }, // he 2
		// Boundary CW: twins of 2, 1, 0.
		{ 2, OptionalIndex::some(2), 4, OptionalIndex::none()  }, // he 3 (0->2)
		{ 1, OptionalIndex::some(1), 5, OptionalIndex::none()  }, // he 4 (2->1)
		{ 0, OptionalIndex::some(0), 3, OptionalIndex::none()  }, // he 5 (1->0)
	};
	return m;
}

// Build the slice-1 quad: 4 verts, 1 face, 8 halfedges.
static HalfedgeMesh build_quad() {
	HalfedgeMesh m;
	m.vertex_count = 4;
	m.face_count = 1;
	m.halfedges = {
		{ 1, OptionalIndex::some(7), 1, OptionalIndex::some(0) }, // 0->1
		{ 2, OptionalIndex::some(6), 2, OptionalIndex::some(0) }, // 1->2
		{ 3, OptionalIndex::some(5), 3, OptionalIndex::some(0) }, // 2->3
		{ 0, OptionalIndex::some(4), 0, OptionalIndex::some(0) }, // 3->0
		{ 3, OptionalIndex::some(3), 5, OptionalIndex::none()  }, // 0->3
		{ 2, OptionalIndex::some(2), 6, OptionalIndex::none()  }, // 3->2
		{ 1, OptionalIndex::some(1), 7, OptionalIndex::none()  }, // 2->1
		{ 0, OptionalIndex::some(0), 4, OptionalIndex::none()  }, // 1->0
	};
	return m;
}

int main() {
	bool ok = true;

	ok &= rc::check("triangle has 6 halfedges", [] {
		HalfedgeMesh m = build_triangle();
		RC_ASSERT(m.he_count() == 6);
	});

	ok &= rc::check("triangle is manifold", [] {
		HalfedgeMesh m = build_triangle();
		RC_ASSERT(is_manifold(m));
	});

	ok &= rc::check("quad has 8 halfedges", [] {
		HalfedgeMesh m = build_quad();
		RC_ASSERT(m.he_count() == 8);
	});

	ok &= rc::check("quad is manifold", [] {
		HalfedgeMesh m = build_quad();
		RC_ASSERT(is_manifold(m));
	});

	// Mutating the triangle to break twin-involution should be detected.
	ok &= rc::check("manifold? rejects broken twin involution", [] {
		HalfedgeMesh m = build_triangle();
		m.halfedges[0].twin = OptionalIndex::some(4); // mismatch (was 5)
		RC_ASSERT(!is_manifold(m));
	});

	// Euler characteristic V - E + F = 2 (counting outer face).
	ok &= rc::check("triangle Euler V-E+F = 2", [] {
		HalfedgeMesh m = build_triangle();
		const std::size_t V = m.vertex_count;
		const std::size_t F = m.face_count + 1; // + outer face
		const std::size_t E = m.he_count() / 2;
		RC_ASSERT(V + F - E == 2);
	});

	return ok ? 0 : 1;
}
