// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
#ifndef CURVENET_SCALED_FRAMES_H
#define CURVENET_SCALED_FRAMES_H

// C++ mirror of `lean/Curvenet/ScaledFrames.lean`. 3x3 matrix toolbox
// + smallest-rotation + isolated-segment deformation gradient (slice 4)
// + 3x3 determinant / inverse / `F = (B S)·(B̆ S̆)⁻¹` (slice 15).

#include <cmath>
#include <cstddef>
#include <vector>

#include "vec3.h"

namespace curvenet {
namespace scaled_frames {

// Row-major 3x3 stored as a 9-element vector. Column j of row i lives at
// index i*3 + j.
using Mat3 = std::vector<double>;

inline double mat3_get(const Mat3 &m, std::size_t i, std::size_t j) {
	return m[i * 3 + j];
}

inline Mat3 mat3_make(double m00, double m01, double m02,
                       double m10, double m11, double m12,
                       double m20, double m21, double m22) {
	return { m00, m01, m02, m10, m11, m12, m20, m21, m22 };
}

inline Mat3 mat3_identity() {
	return mat3_make(1.0, 0.0, 0.0,
	                 0.0, 1.0, 0.0,
	                 0.0, 0.0, 1.0);
}

inline Mat3 mat3_add(const Mat3 &a, const Mat3 &b) {
	Mat3 out(9, 0.0);
	for (std::size_t k = 0; k < 9; ++k) {
		out[k] = a[k] + b[k];
	}
	return out;
}

inline Mat3 mat3_smul(double s, const Mat3 &a) {
	Mat3 out(9, 0.0);
	for (std::size_t k = 0; k < 9; ++k) {
		out[k] = s * a[k];
	}
	return out;
}

inline Mat3 mat3_mul(const Mat3 &a, const Mat3 &b) {
	Mat3 out(9, 0.0);
	for (std::size_t i = 0; i < 3; ++i) {
		for (std::size_t j = 0; j < 3; ++j) {
			out[i * 3 + j] = mat3_get(a, i, 0) * mat3_get(b, 0, j) +
			                 mat3_get(a, i, 1) * mat3_get(b, 1, j) +
			                 mat3_get(a, i, 2) * mat3_get(b, 2, j);
		}
	}
	return out;
}

inline Vec3 mat3_mul_vec(const Mat3 &m, const Vec3 &v) {
	return { mat3_get(m, 0, 0) * v.x + mat3_get(m, 0, 1) * v.y + mat3_get(m, 0, 2) * v.z,
	         mat3_get(m, 1, 0) * v.x + mat3_get(m, 1, 1) * v.y + mat3_get(m, 1, 2) * v.z,
	         mat3_get(m, 2, 0) * v.x + mat3_get(m, 2, 1) * v.y + mat3_get(m, 2, 2) * v.z };
}

inline Mat3 skew(const Vec3 &v) {
	return mat3_make( 0.0, -v.z,   v.y,
	                  v.z,  0.0,  -v.x,
	                 -v.y,  v.x,   0.0);
}

// Tangent (unit) and length of a segment from `p` to `q`.
inline std::pair<Vec3, double> tangent_length(const Vec3 &p, const Vec3 &q) {
	const Vec3 d = q - p;
	const double l = d.length();
	const double inv = (l == 0.0) ? 0.0 : 1.0 / l;
	return { Vec3{ d.x * inv, d.y * inv, d.z * inv }, l };
}

// Smallest rotation aligning unit vector `from_v` to unit vector `to_v`.
// R = I + K + K² / (1 + cos θ),   K = skew(from_v × to_v).
inline Mat3 smallest_rotation(const Vec3 &from_v, const Vec3 &to_v) {
	const Vec3 cross_v = { from_v.y * to_v.z - from_v.z * to_v.y,
	                        from_v.z * to_v.x - from_v.x * to_v.z,
	                        from_v.x * to_v.y - from_v.y * to_v.x };
	const double cos_t = from_v.x * to_v.x + from_v.y * to_v.y + from_v.z * to_v.z;
	const Mat3 K  = skew(cross_v);
	const Mat3 K2 = mat3_mul(K, K);
	const double denom = 1.0 + cos_t;
	const double inv_denom = (denom == 0.0) ? 0.0 : 1.0 / denom;
	return mat3_add(mat3_add(mat3_identity(), K), mat3_smul(inv_denom, K2));
}

// Isolated-curve segment deformation gradient: F = (l/l̆) · R(t̆, t).
inline Mat3 isolated_segment_gradient(const Vec3 &rest_p, const Vec3 &rest_q,
                                       const Vec3 &posed_p, const Vec3 &posed_q) {
	const std::pair<Vec3, double> rt = tangent_length(rest_p,  rest_q);
	const std::pair<Vec3, double> pt = tangent_length(posed_p, posed_q);
	const double r = (rt.second == 0.0) ? 0.0 : pt.second / rt.second;
	return mat3_smul(r, smallest_rotation(rt.first, pt.first));
}

inline double mat3_det(const Mat3 &m) {
	const double a = mat3_get(m, 0, 0);
	const double b = mat3_get(m, 0, 1);
	const double c = mat3_get(m, 0, 2);
	const double d = mat3_get(m, 1, 0);
	const double e = mat3_get(m, 1, 1);
	const double f = mat3_get(m, 1, 2);
	const double g = mat3_get(m, 2, 0);
	const double h = mat3_get(m, 2, 1);
	const double i = mat3_get(m, 2, 2);
	return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
}

inline Mat3 mat3_inv(const Mat3 &m) {
	const double a = mat3_get(m, 0, 0);
	const double b = mat3_get(m, 0, 1);
	const double c = mat3_get(m, 0, 2);
	const double d = mat3_get(m, 1, 0);
	const double e = mat3_get(m, 1, 1);
	const double f = mat3_get(m, 1, 2);
	const double g = mat3_get(m, 2, 0);
	const double h = mat3_get(m, 2, 1);
	const double i = mat3_get(m, 2, 2);
	const double det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
	const double inv = (det == 0.0) ? 0.0 : 1.0 / det;
	return mat3_make(
		inv * (e * i - f * h), inv * (c * h - b * i), inv * (b * f - c * e),
		inv * (f * g - d * i), inv * (a * i - c * g), inv * (c * d - a * f),
		inv * (d * h - e * g), inv * (b * g - a * h), inv * (a * e - b * d));
}

inline Mat3 deformation_gradient(const Mat3 &rest_frame, const Mat3 &posed_frame) {
	return mat3_mul(posed_frame, mat3_inv(rest_frame));
}

inline bool mat3_within_eps(const Mat3 &a, const Mat3 &b, double eps) {
	for (std::size_t k = 0; k < 9; ++k) {
		if (std::abs(a[k] - b[k]) >= eps) {
			return false;
		}
	}
	return true;
}

} // namespace scaled_frames
} // namespace curvenet

#endif
