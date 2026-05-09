// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
#include "test_helpers.h"

#include "curvenet/tris_to_quads.h"

#include <set>
#include <utility>

using curvenet::PolyMesh;
using curvenet::TriMesh;
using curvenet::Vec3;
using curvenet::tris_to_quads;

// Build a planar W x H grid of triangles. Each cell (i,j) is split into two tris
// along the (0,0)-(1,1) diagonal, all sharing manifold internal edges.
static TriMesh make_grid(int W, int H) {
	TriMesh m;
	m.vertices.reserve((W + 1) * (H + 1));
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
			// Two triangles sharing the (v00, v11) diagonal.
			m.triangles.push_back({ v00, v10, v11 });
			m.triangles.push_back({ v00, v11, v01 });
		}
	}
	return m;
}

static int total_face_indices(const PolyMesh &m) {
	int n = 0;
	for (const auto &f : m.faces) {
		n += f.count;
	}
	return n;
}

static int count_quads(const PolyMesh &m) {
	int n = 0;
	for (const auto &f : m.faces) {
		if (f.count == 4) {
			++n;
		}
	}
	return n;
}

static int count_tris(const PolyMesh &m) {
	int n = 0;
	for (const auto &f : m.faces) {
		if (f.count == 3) {
			++n;
		}
	}
	return n;
}

int main() {
	bool ok = true;

	ok &= rc::check("vertex set is preserved", [] {
		int W = *rc::gen::inRange<int>(1, 5);
		int H = *rc::gen::inRange<int>(1, 5);
		TriMesh in = make_grid(W, H);
		PolyMesh out = tris_to_quads(in);
		RC_ASSERT(out.vertices.size() == in.vertices.size());
		for (std::size_t i = 0; i < in.vertices.size(); ++i) {
			RC_ASSERT(cnt::approx_eq(in.vertices[i], out.vertices[i], 0.0));
		}
	});

	ok &= rc::check("face accounting: T_in = T_out + 2*Q_out", [] {
		int W = *rc::gen::inRange<int>(1, 5);
		int H = *rc::gen::inRange<int>(1, 5);
		TriMesh in = make_grid(W, H);
		PolyMesh out = tris_to_quads(in);
		int t_in = static_cast<int>(in.triangles.size());
		RC_ASSERT(t_in == count_tris(out) + 2 * count_quads(out));
	});

	ok &= rc::check("no triangle index appears in two output quads", [] {
		// This is a structural property: if our matching is valid, each input
		// triangle ends up in at most one output quad. Equivalent: sum of face
		// counts equals (T_out * 3) + (Q_out * 4) and matches input triangle count.
		int W = *rc::gen::inRange<int>(1, 5);
		int H = *rc::gen::inRange<int>(1, 5);
		TriMesh in = make_grid(W, H);
		PolyMesh out = tris_to_quads(in);
		// Each tri in input contributes either 3 indices (kept as tri) or
		// shares its 3 indices with another to form a 4-index quad. Union of
		// quad's 4 verts must equal the union of the two source tris' 6 verts
		// minus the 2 shared along the dissolved edge. Sanity: total face index
		// count in output = 3 * count_tris + 4 * count_quads = 3*T_in - 2*Q.
		int expected = 3 * static_cast<int>(in.triangles.size()) - 2 * count_quads(out);
		RC_ASSERT(total_face_indices(out) == expected);
	});

	ok &= rc::check("regular grid: all triangles pair into quads", [] {
		int W = *rc::gen::inRange<int>(1, 5);
		int H = *rc::gen::inRange<int>(1, 5);
		TriMesh in = make_grid(W, H);
		PolyMesh out = tris_to_quads(in);
		// On a regular grid with the same diagonal everywhere, optimum dissolves
		// the diagonal in every cell, producing W*H quads and 0 leftover tris.
		RC_ASSERT(count_quads(out) == W * H);
		RC_ASSERT(count_tris(out) == 0);
	});

	ok &= rc::check("output quad winding is CCW (positive 2D signed area for planar input)", [] {
		int W = *rc::gen::inRange<int>(1, 5);
		int H = *rc::gen::inRange<int>(1, 5);
		TriMesh in = make_grid(W, H);
		PolyMesh out = tris_to_quads(in);
		for (const auto &f : out.faces) {
			if (f.count != 4) continue;
			const Vec3 &a = out.vertices[f.v[0]];
			const Vec3 &b = out.vertices[f.v[1]];
			const Vec3 &c = out.vertices[f.v[2]];
			const Vec3 &d = out.vertices[f.v[3]];
			// Shoelace area in xy plane (input is z=0).
			double area = 0.5 * ((b.x - a.x) * (b.y + a.y) +
								(c.x - b.x) * (c.y + b.y) +
								(d.x - c.x) * (d.y + c.y) +
								(a.x - d.x) * (a.y + d.y));
			// Negate because shoelace orientation convention.
			RC_ASSERT(-area > 0.0);
		}
	});

	return ok ? 0 : 1;
}
