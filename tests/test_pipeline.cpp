// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
//
// End-to-end pipeline test: TriMesh -> tris_to_quads -> bind_polymesh ->
// CurveNet build (straight-line cubic boundaries) -> deform. Exercises every
// math layer the Godot deformer uses.

#include "test_helpers.h"

#include "curvenet/binding.h"
#include "curvenet/curvenet.h"
#include "curvenet/tris_to_quads.h"

using curvenet::BoundaryCurve;
using curvenet::BoundFace;
using curvenet::BoundVertex;
using curvenet::CurveNet;
using curvenet::PolyMesh;
using curvenet::TriMesh;
using curvenet::Vec3;
using curvenet::bind_polymesh;
using curvenet::deform;
using curvenet::tris_to_quads;

// Build a planar W x H grid of triangles (same as test_tris_to_quads).
static TriMesh make_grid(int W, int H) {
	TriMesh m;
	for (int y = 0; y <= H; ++y) {
		for (int x = 0; x <= W; ++x) {
			m.vertices.push_back(Vec3{ static_cast<double>(x), static_cast<double>(y), 0.0 });
		}
	}
	auto idx = [W](int x, int y) { return y * (W + 1) + x; };
	for (int y = 0; y < H; ++y) {
		for (int x = 0; x < W; ++x) {
			int v00 = idx(x, y);
			int v10 = idx(x + 1, y);
			int v01 = idx(x, y + 1);
			int v11 = idx(x + 1, y + 1);
			m.triangles.push_back({ v00, v10, v11 });
			m.triangles.push_back({ v00, v11, v01 });
		}
	}
	return m;
}

// Straight-line cubic Bezier between two points (control points at thirds).
static BoundaryCurve straight(const Vec3 &a, const Vec3 &b) {
	BoundaryCurve c;
	c.c0 = a;
	c.c3 = b;
	c.c1 = a + (b - a) * (1.0 / 3.0);
	c.c2 = a + (b - a) * (2.0 / 3.0);
	return c;
}

static CurveNet identity_curvenet(const PolyMesh &mesh) {
	CurveNet net;
	net.faces.reserve(mesh.faces.size());
	for (const auto &f : mesh.faces) {
		BoundFace bf;
		if (f.count == 4) {
			const Vec3 &v00 = mesh.vertices[f.v[0]];
			const Vec3 &v10 = mesh.vertices[f.v[1]];
			const Vec3 &v11 = mesh.vertices[f.v[2]];
			const Vec3 &v01 = mesh.vertices[f.v[3]];
			bf.boundaries.push_back(straight(v00, v10));
			bf.boundaries.push_back(straight(v10, v11));
			bf.boundaries.push_back(straight(v11, v01));
			bf.boundaries.push_back(straight(v01, v00));
		}
		net.faces.push_back(bf);
	}
	return net;
}

int main() {
	bool ok = true;

	ok &= rc::check("identity pipeline: deform with straight boundaries returns input", [] {
		int W = *rc::gen::inRange<int>(1, 4);
		int H = *rc::gen::inRange<int>(1, 4);
		TriMesh in = make_grid(W, H);
		PolyMesh poly = tris_to_quads(in);
		auto bindings = bind_polymesh(poly);
		CurveNet net = identity_curvenet(poly);
		auto out = deform(net, bindings);
		// Each bound vertex should map back to its original position.
		for (std::size_t i = 0; i < poly.vertices.size(); ++i) {
			if (bindings[i].face_index < 0) {
				continue; // fell on a triangle face
			}
			RC_ASSERT(cnt::approx_eq(out[i], poly.vertices[i], 1e-6));
		}
	});

	ok &= rc::check("translation pipeline: shift all curves by d -> all bound verts shift by d", [](Vec3 d) {
		int W = *rc::gen::inRange<int>(1, 4);
		int H = *rc::gen::inRange<int>(1, 4);
		TriMesh in = make_grid(W, H);
		PolyMesh poly = tris_to_quads(in);
		auto bindings = bind_polymesh(poly);
		CurveNet net = identity_curvenet(poly);
		// Translate every boundary by d.
		CurveNet net2 = net;
		for (auto &face : net2.faces) {
			for (auto &b : face.boundaries) {
				b.c0 = b.c0 + d;
				b.c1 = b.c1 + d;
				b.c2 = b.c2 + d;
				b.c3 = b.c3 + d;
			}
		}
		auto a = deform(net, bindings);
		auto b = deform(net2, bindings);
		for (std::size_t i = 0; i < poly.vertices.size(); ++i) {
			if (bindings[i].face_index < 0) {
				continue;
			}
			RC_ASSERT(cnt::approx_eq(a[i] + d, b[i], 1e-9));
		}
	});

	ok &= rc::check("face accounting after pipeline: every quad has 4 corner verts bound to it", [] {
		int W = *rc::gen::inRange<int>(2, 4);
		int H = *rc::gen::inRange<int>(2, 4);
		TriMesh in = make_grid(W, H);
		PolyMesh poly = tris_to_quads(in);
		auto bindings = bind_polymesh(poly);
		// On a regular grid, every triangle pair becomes a quad, so every
		// vertex binds to a quad (face_index >= 0).
		int unbound = 0;
		for (const auto &b : bindings) {
			if (b.face_index < 0) ++unbound;
		}
		RC_ASSERT(unbound == 0);
	});

	return ok ? 0 : 1;
}
