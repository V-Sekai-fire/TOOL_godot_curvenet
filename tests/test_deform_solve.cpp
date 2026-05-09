// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
//
// Mirrors `lean/Curvenet/HarmonicSolve.lean` + `lean/Curvenet/DeformSolve.lean`
// instance theorems (slices 7, 8, 9). End-to-end property: a sample
// translation by (5, 0, 0) on triangleWithSample produces pure
// translation at every unpromoted vertex.
#include "test_helpers.h"

#include "curvenet/cut_algorithm.h"
#include "curvenet/cut_mesh.h"
#include "curvenet/deform_solve.h"
#include "curvenet/halfedge.h"
#include "curvenet/harmonic_solve.h"

#include <cmath>
#include <cstddef>
#include <vector>

using curvenet::Halfedge;
using curvenet::HalfedgeMesh;
using curvenet::OptionalIndex;
using curvenet::Vec3;
namespace cm = curvenet::cut_mesh;
namespace hs = curvenet::harmonic_solve;
namespace ds = curvenet::deform_solve;

static int single_sample_col(int, int, bool) { return 0; }

static cm::CutMesh build_triangle_with_sample() {
	HalfedgeMesh base;
	base.vertex_count = 3;
	base.face_count = 1;
	base.halfedges = {
		{ 1, OptionalIndex::some(5), 1, OptionalIndex::some(0) },
		{ 2, OptionalIndex::some(4), 2, OptionalIndex::some(0) },
		{ 0, OptionalIndex::some(3), 0, OptionalIndex::some(0) },
		{ 2, OptionalIndex::some(2), 4, OptionalIndex::none()  },
		{ 1, OptionalIndex::some(1), 5, OptionalIndex::none()  },
		{ 0, OptionalIndex::some(0), 3, OptionalIndex::none()  },
	};
	cm::CutMesh out;
	out.base = base;
	out.vertex_kind = {
		cm::CutVertexKind::sample_kind(0, 0, false),
		cm::CutVertexKind::mesh_vertex_kind(),
		cm::CutVertexKind::mesh_vertex_kind() };
	out.segment_of_halfedge.assign(6, -1);
	return out;
}

static std::vector<Vec3> equilateral_triangle_positions() {
	return { { 0.0, 0.0, 0.0 },
	         { 1.0, 0.0, 0.0 },
	         { 0.5, 0.0, 0.8660254037844386 } };
}

int main() {
	bool ok = true;

	const cm::CutMesh m  = build_triangle_with_sample();
	const std::vector<Vec3> positions = equilateral_triangle_positions();

	// --- HarmonicSolve slice 7 ---
	ok &= rc::check("scalar harmonic: fc = [5] -> vertices 1, 2 = 5", [m, positions] {
		const std::vector<double> fc = { 5.0 };
		const std::vector<double> xv =
			hs::solve_scalar(m, positions, single_sample_col, fc);
		RC_ASSERT(std::abs(xv[1] - 5.0) < 1e-9);
		RC_ASSERT(std::abs(xv[2] - 5.0) < 1e-9);
	});

	// --- HarmonicSolve slice 8 (multi-column) ---
	ok &= rc::check("multi-column: 3 RHS columns yield 3-vector at each vertex", [m, positions] {
		// Fc is 1 × 3.
		const std::vector<double> Fc = { 5.0, 10.0, 15.0 };
		const std::vector<double> Xv =
			hs::solve_multi(m, positions, single_sample_col, Fc, 3);
		// Xv is 3 × 3. Vertex 1 row should be (5, 10, 15).
		RC_ASSERT(std::abs(Xv[1 * 3 + 0] - 5.0)  < 1e-9);
		RC_ASSERT(std::abs(Xv[1 * 3 + 1] - 10.0) < 1e-9);
		RC_ASSERT(std::abs(Xv[1 * 3 + 2] - 15.0) < 1e-9);
	});

	// --- DeformSolve slice 9 (full §4.3) ---
	ok &= rc::check("full deform: identity F + Xc = (5,0,0) -> pure translation by 5", [m, positions] {
		// Fc is 1 × 9 with row-major identity.
		const std::vector<double> Fc = {
			1.0, 0.0, 0.0,
			0.0, 1.0, 0.0,
			0.0, 0.0, 1.0 };
		// Xc is 1 × 3 with sample target (5, 0, 0).
		const std::vector<double> Xc = { 5.0, 0.0, 0.0 };
		const std::vector<double> Xv =
			ds::solve_deformation(m, positions, single_sample_col, Fc, Xc);
		// Vertex 1 (rest at (1,0,0)) should land at (6, 0, 0).
		RC_ASSERT(std::abs(Xv[1 * 3 + 0] - 6.0) < 1e-9);
		RC_ASSERT(std::abs(Xv[1 * 3 + 1])       < 1e-9);
		RC_ASSERT(std::abs(Xv[1 * 3 + 2])       < 1e-9);
		// Vertex 2 (rest at (0.5, 0, sqrt(3)/2)) should land at
		// (5.5, 0, sqrt(3)/2).
		RC_ASSERT(std::abs(Xv[2 * 3 + 0] - 5.5) < 1e-9);
		RC_ASSERT(std::abs(Xv[2 * 3 + 1])       < 1e-9);
		RC_ASSERT(std::abs(Xv[2 * 3 + 2] - 0.8660254037844386) < 1e-9);
	});

	return ok ? 0 : 1;
}
