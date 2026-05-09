// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
#ifndef CURVENET_BILINEAR_INVERSE_H
#define CURVENET_BILINEAR_INVERSE_H

#include "vec3.h"

#include <limits>

namespace curvenet {

// Result of inverting bilinear interpolation: (s,t) ∈ [0,1]^2 plus the
// orthogonal residual after the inversion (distance from query point V to
// the bilinear surface).
struct BilinearInverseResult {
	double s = 0.0;
	double t = 0.0;
	double residual = 0.0;
	bool converged = false;
};

// Solve V ≈ (1-s)(1-t)*P00 + s(1-t)*P10 + s*t*P11 + (1-s)*t*P01 via Gauss-Newton.
// CCW corner order: P00, P10, P11, P01.
//
// For planar parallelograms a single Newton step is exact; for warped quads ~5
// iterations bring the residual to machine precision. For points outside the
// patch domain, (s,t) may exit [0,1] — caller decides whether to clamp or
// reject. We never clamp internally so the residual reflects the true distance.
inline BilinearInverseResult solve_bilinear_inverse(
		const Vec3 &V, const Vec3 &P00, const Vec3 &P10, const Vec3 &P11, const Vec3 &P01,
		int max_iter = 16, double tol = 1e-12) {
	double s = 0.5;
	double t = 0.5;
	BilinearInverseResult out;

	auto patch_at = [&](double s_, double t_) {
		const double ms = 1.0 - s_;
		const double mt = 1.0 - t_;
		return P00 * (ms * mt) + P10 * (s_ * mt) + P11 * (s_ * t_) + P01 * (ms * t_);
	};
	auto dpatch_ds = [&](double t_) {
		// (1-t)(P10-P00) + t(P11-P01)
		return (P10 - P00) * (1.0 - t_) + (P11 - P01) * t_;
	};
	auto dpatch_dt = [&](double s_) {
		// (1-s)(P01-P00) + s(P11-P10)
		return (P01 - P00) * (1.0 - s_) + (P11 - P10) * s_;
	};

	for (int iter = 0; iter < max_iter; ++iter) {
		Vec3 r = V - patch_at(s, t);
		double rlen2 = r.dot(r);
		if (rlen2 < tol * tol) {
			out.converged = true;
			break;
		}
		Vec3 ds = dpatch_ds(t);
		Vec3 dt = dpatch_dt(s);
		// Gauss-Newton: solve [JᵀJ] δ = Jᵀr where J = [ds | dt] is 3×2.
		double a = ds.dot(ds);
		double b = ds.dot(dt);
		double c = dt.dot(dt);
		double det = a * c - b * b;
		if (det == 0.0) {
			// Degenerate quad (collinear corners); abort with infinite residual
			// so callers (e.g. bind_polymesh) skip this face when picking owner.
			out.s = s;
			out.t = t;
			out.residual = std::numeric_limits<double>::infinity();
			out.converged = false;
			return out;
		}
		double inv_det = 1.0 / det;
		double rs = ds.dot(r);
		double rt = dt.dot(r);
		double delta_s = (c * rs - b * rt) * inv_det;
		double delta_t = (-b * rs + a * rt) * inv_det;
		s += delta_s;
		t += delta_t;
		if (delta_s * delta_s + delta_t * delta_t < tol * tol) {
			out.converged = true;
			break;
		}
	}

	out.s = s;
	out.t = t;
	{
		Vec3 r = V - patch_at(s, t);
		out.residual = r.length();
	}
	return out;
}

} // namespace curvenet

#endif
