// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
//
// Mirrors `lean/Curvenet/PolygonLaplacian.lean`'s instance theorems.
#include "test_helpers.h"

#include "curvenet/polygon_laplacian.h"

#include <cmath>
#include <vector>

using curvenet::Vec3;
namespace pl = curvenet::polygon_laplacian;

static std::vector<Vec3> equilateral_triangle() {
	return { { 0.0, 0.0, 0.0 },
	         { 1.0, 0.0, 0.0 },
	         { 0.5, 0.0, 0.8660254037844386 } };
}

static std::vector<Vec3> unit_square() {
	return { { 0.0, 0.0, 0.0 },
	         { 1.0, 0.0, 0.0 },
	         { 1.0, 0.0, 1.0 },
	         { 0.0, 0.0, 1.0 } };
}

int main() {
	bool ok = true;

	ok &= rc::check("triangle Laplacian: row sums vanish", [] {
		const std::vector<Vec3> v = equilateral_triangle();
		const std::vector<double> L = pl::triangle_cot_laplacian(v[0], v[1], v[2]);
		RC_ASSERT(pl::row_sums_within(L, 3, 1e-12));
	});

	ok &= rc::check("triangle Laplacian: symmetric", [] {
		const std::vector<Vec3> v = equilateral_triangle();
		const std::vector<double> L = pl::triangle_cot_laplacian(v[0], v[1], v[2]);
		RC_ASSERT(pl::is_symmetric_within(L, 3, 1e-12));
	});

	ok &= rc::check("equilateral triangle: off-diagonal = -1/(2 sqrt 3)", [] {
		const std::vector<Vec3> v = equilateral_triangle();
		const std::vector<double> L = pl::triangle_cot_laplacian(v[0], v[1], v[2]);
		const double expected = -1.0 / (2.0 * std::sqrt(3.0));
		RC_ASSERT(std::abs(pl::get_at(L, 3, 0, 1) - expected) < 1e-12);
	});

	ok &= rc::check("quad Laplacian (fan): row sums vanish", [] {
		const std::vector<Vec3> v = unit_square();
		const std::vector<double> L = pl::polygon_cot_laplacian(v);
		RC_ASSERT(pl::row_sums_within(L, 4, 1e-12));
	});

	ok &= rc::check("quad Laplacian (fan): symmetric", [] {
		const std::vector<Vec3> v = unit_square();
		const std::vector<double> L = pl::polygon_cot_laplacian(v);
		RC_ASSERT(pl::is_symmetric_within(L, 4, 1e-12));
	});

	return ok ? 0 : 1;
}
