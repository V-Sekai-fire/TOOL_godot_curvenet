// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
#ifndef CURVENET_POLYGON_LAPLACIAN_H
#define CURVENET_POLYGON_LAPLACIAN_H

// C++ mirror of `lean/Curvenet/PolygonLaplacian.lean`. Cotangent
// Laplacian for triangles + fan-triangulation generalisation to
// arbitrary polygons. Used by the cut-mesh Laplacian assembly.

#include <cstddef>
#include <vector>

#include "vec3.h"

namespace curvenet {
namespace polygon_laplacian {

// Cotangent of the triangle's interior angle at apex:
//   cot θ = (a · b) / |a × b|
// where a = other1 - apex, b = other2 - apex.
inline double cot_at_vertex(const Vec3 &apex, const Vec3 &other1, const Vec3 &other2) {
	const Vec3 a = other1 - apex;
	const Vec3 b = other2 - apex;
	const double dot = a.dot(b);
	const Vec3 c = { a.y * b.z - a.z * b.y,
	                 a.z * b.x - a.x * b.z,
	                 a.x * b.y - a.y * b.x };
	const double cross_len = c.length();
	return dot / cross_len;
}

// Row-major n×n matrix as a flat vector.
inline double get_at(const std::vector<double> &m, std::size_t n, std::size_t i, std::size_t j) {
	return m[i * n + j];
}
inline void   set_at(std::vector<double> &m, std::size_t n, std::size_t i, std::size_t j, double v) {
	m[i * n + j] = v;
}
inline void   add_at(std::vector<double> &m, std::size_t n, std::size_t i, std::size_t j, double v) {
	m[i * n + j] += v;
}

// Cotangent Laplacian for a single triangle (3×3 row-major).
inline std::vector<double> triangle_cot_laplacian(const Vec3 &a, const Vec3 &b, const Vec3 &c) {
	const double cot_a = cot_at_vertex(a, b, c);
	const double cot_b = cot_at_vertex(b, a, c);
	const double cot_c = cot_at_vertex(c, a, b);
	const double w_bc = cot_a * 0.5;
	const double w_ac = cot_b * 0.5;
	const double w_ab = cot_c * 0.5;

	std::vector<double> m(9, 0.0);
	set_at(m, 3, 0, 1, -w_ab); set_at(m, 3, 1, 0, -w_ab);
	set_at(m, 3, 0, 2, -w_ac); set_at(m, 3, 2, 0, -w_ac);
	set_at(m, 3, 1, 2, -w_bc); set_at(m, 3, 2, 1, -w_bc);
	set_at(m, 3, 0, 0, w_ab + w_ac);
	set_at(m, 3, 1, 1, w_ab + w_bc);
	set_at(m, 3, 2, 2, w_ac + w_bc);
	return m;
}

// Polygonal cotangent Laplacian by fan triangulation from vertex 0:
// triangles (0, i, i+1) for i = 1..n-2. Each triangle contributes its
// 3×3 Laplacian to the n×n polygon matrix at the corresponding indices.
inline std::vector<double> polygon_cot_laplacian(const std::vector<Vec3> &poly) {
	const std::size_t n = poly.size();
	std::vector<double> m(n * n, 0.0);
	if (n < 3) {
		return m;
	}
	const Vec3 v0 = poly[0];
	for (std::size_t i = 1; i + 1 < n; ++i) {
		const Vec3 vi = poly[i];
		const Vec3 vj = poly[i + 1];
		const std::vector<double> tri = triangle_cot_laplacian(v0, vi, vj);
		const std::size_t idx[3] = { 0, i, i + 1 };
		for (std::size_t li = 0; li < 3; ++li) {
			for (std::size_t lj = 0; lj < 3; ++lj) {
				add_at(m, n, idx[li], idx[lj], get_at(tri, 3, li, lj));
			}
		}
	}
	return m;
}

// |row sum| < eps for every row.
inline bool row_sums_within(const std::vector<double> &m, std::size_t n, double eps) {
	for (std::size_t i = 0; i < n; ++i) {
		double s = 0.0;
		for (std::size_t j = 0; j < n; ++j) {
			s += get_at(m, n, i, j);
		}
		if (std::abs(s) >= eps) {
			return false;
		}
	}
	return true;
}

// |m[i,j] - m[j,i]| < eps everywhere.
inline bool is_symmetric_within(const std::vector<double> &m, std::size_t n, double eps) {
	for (std::size_t i = 0; i < n; ++i) {
		for (std::size_t j = i + 1; j < n; ++j) {
			if (std::abs(get_at(m, n, i, j) - get_at(m, n, j, i)) >= eps) {
				return false;
			}
		}
	}
	return true;
}

} // namespace polygon_laplacian
} // namespace curvenet

#endif
