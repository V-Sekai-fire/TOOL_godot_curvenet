// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
#include "test_helpers.h"

#include "curvenet/binding.h"

using curvenet::BoundVertex;
using curvenet::PolyFace;
using curvenet::PolyMesh;
using curvenet::Vec3;
using curvenet::bind_polymesh;

// A 1-quad PolyMesh for testing.
static PolyMesh make_unit_quad_mesh() {
	PolyMesh m;
	m.vertices = { { 0.0, 0.0, 0.0 }, { 1.0, 0.0, 0.0 }, { 1.0, 1.0, 0.0 }, { 0.0, 1.0, 0.0 } };
	PolyFace f;
	f.count = 4;
	f.v[0] = 0;
	f.v[1] = 1;
	f.v[2] = 2;
	f.v[3] = 3;
	m.faces.push_back(f);
	return m;
}

int main() {
	bool ok = true;

	ok &= rc::check("each vertex of a unit quad mesh binds to face 0", [] {
		PolyMesh m = make_unit_quad_mesh();
		auto bindings = bind_polymesh(m);
		RC_ASSERT(bindings.size() == m.vertices.size());
		for (const auto &b : bindings) {
			RC_ASSERT(b.face_index == 0);
		}
	});

	ok &= rc::check("corner vertices bind at corner (s,t)", [] {
		PolyMesh m = make_unit_quad_mesh();
		auto b = bind_polymesh(m);
		RC_ASSERT(std::fabs(b[0].s) < 1e-6 && std::fabs(b[0].t) < 1e-6); // P00 -> (0,0)
		RC_ASSERT(std::fabs(b[1].s - 1.0) < 1e-6 && std::fabs(b[1].t) < 1e-6); // P10 -> (1,0)
		RC_ASSERT(std::fabs(b[2].s - 1.0) < 1e-6 && std::fabs(b[2].t - 1.0) < 1e-6); // P11 -> (1,1)
		RC_ASSERT(std::fabs(b[3].s) < 1e-6 && std::fabs(b[3].t - 1.0) < 1e-6); // P01 -> (0,1)
	});

	ok &= rc::check("interior vertex binds with small residual", [] {
		PolyMesh m = make_unit_quad_mesh();
		// Add a vertex at the center.
		m.vertices.push_back(Vec3{ 0.5, 0.5, 0.0 });
		auto b = bind_polymesh(m);
		RC_ASSERT(std::fabs(b[4].s - 0.5) < 1e-6);
		RC_ASSERT(std::fabs(b[4].t - 0.5) < 1e-6);
	});

	ok &= rc::check("triangle-only mesh leaves all bindings as face_index = -1", [] {
		// 4 vertices, 1 triangle face, 0 quads.
		PolyMesh m = make_unit_quad_mesh();
		m.faces.clear();
		PolyFace tri;
		tri.count = 3;
		tri.v[0] = 0;
		tri.v[1] = 1;
		tri.v[2] = 2;
		m.faces.push_back(tri);
		auto b = bind_polymesh(m);
		for (const auto &bv : b) {
			RC_ASSERT(bv.face_index == -1);
		}
	});

	return ok ? 0 : 1;
}
