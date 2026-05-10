// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
//
// Curvenet-driven Direct Delta Mush (Le & Lewis 2019, adapted).
// C++ mirror of `lean/Curvenet/DirectDeltaMush.lean` and
// `lean/Curvenet/DirectDeltaMushBind.lean`.
//
// Runtime kernel (used per drag):
//   pos[v] = sum over influences (i, w) of:  w * (T_i * rest_pos[v])
// where T_i is the per-handle 4x4 affine transform (DeGoes22 §3
// scaled-frame ratio between current and rest configurations).
//
// Bind-time pipeline (used at topology / knot change):
//   1. Capture per-vertex influence weights W[v, i] by running
//      DeGoes22's two-stage solve (HarmonicSolve + DeformSolve)
//      for unit perturbations of each handle.
//   2. Smooth W via damped-Jacobi Laplacian iterations.
//   3. Sparsify each row to the top-K entries by |w| and renormalize
//      so the row sums to 1 (partition of unity).
//
// Quest 3 0.8 ms / 12 RHS / 50k-vertex budget is unreachable for the
// per-drag DeGoes22 §6 solve; DDM moves the §6 work to bind time and
// reduces runtime to a sparse linear combination per vertex.

#ifndef CURVENET_DIRECT_DELTA_MUSH_H
#define CURVENET_DIRECT_DELTA_MUSH_H

#include <array>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace curvenet {
namespace direct_delta_mush {

// ===================================================================
// Runtime kernel — mirrors `DirectDeltaMush.lean`.
// ===================================================================

// Row-major 4x4 affine transform.
using Mat4 = std::array<double, 16>;

// Per-vertex sparse influence list: pairs of (handle_idx, weight).
using Influences = std::vector<std::pair<int, double>>;

inline Mat4 mat4_identity() {
	return Mat4{
		1.0, 0.0, 0.0, 0.0,
		0.0, 1.0, 0.0, 0.0,
		0.0, 0.0, 1.0, 0.0,
		0.0, 0.0, 0.0, 1.0
	};
}

inline Mat4 mat4_translation(double tx, double ty, double tz) {
	return Mat4{
		1.0, 0.0, 0.0, tx,
		0.0, 1.0, 0.0, ty,
		0.0, 0.0, 1.0, tz,
		0.0, 0.0, 0.0, 1.0
	};
}

// Apply 4x4 affine to a 3-point. Treats input as homogeneous (x, y, z, 1);
// returns the first three components of T * v.
inline std::array<double, 3>
apply_transform_4x4(const Mat4 &T, double x, double y, double z) {
	return std::array<double, 3>{
		T[0]  * x + T[1]  * y + T[2]  * z + T[3],
		T[4]  * x + T[5]  * y + T[6]  * z + T[7],
		T[8]  * x + T[9]  * y + T[10] * z + T[11]
	};
}

// LBS-flavored linear combination: weighted average of (T_i * rest)
// across the vertex's influences. Sum of weights does not need to be
// exactly 1 — for harmonic-derived weights it's automatically partition
// of unity.
inline std::array<double, 3>
lbs_matvec(const std::vector<Mat4> &transforms,
            const Influences &infl,
            double rx, double ry, double rz) {
	double x = 0.0, y = 0.0, z = 0.0;
	for (const auto &kv : infl) {
		const int i = kv.first;
		const double w = kv.second;
		const auto t = apply_transform_4x4(transforms[i], rx, ry, rz);
		x += w * t[0];
		y += w * t[1];
		z += w * t[2];
	}
	return std::array<double, 3>{ x, y, z };
}

// ===================================================================
// Bind-time post-processing — mirrors `DirectDeltaMushBind.lean`.
// ===================================================================

// Dense per-vertex weight matrix. Outer index = vertex; inner = handle.
using WeightMatrix = std::vector<std::vector<double>>;

// Per-vertex adjacency list (mesh neighbors of each vertex).
using Adjacency = std::vector<std::vector<int>>;

inline double row_sum(const std::vector<double> &row) {
	double s = 0.0;
	for (double v : row) s += v;
	return s;
}

inline bool fclose(double a, double b, double eps) {
	return std::fabs(a - b) < eps;
}

// Every row sum is within `eps` of 1.0.
inline bool partition_of_unity(const WeightMatrix &W, double eps) {
	for (const auto &row : W) {
		if (!fclose(row_sum(row), 1.0, eps)) return false;
	}
	return true;
}

// Damped-Jacobi Laplacian smoothing on the weight matrix. For each
// pass and each handle column, every vertex's weight is replaced by
//   (1 - omega) * w[v] + omega * mean of w[nbr] over v's neighbors.
// Runs `nu` passes. Uniform mean weighting — the harmonic prior
// (from step 1's §6 solve) already used cot weights, so this step is
// local diffusion only. omega = 0.5 mixes evenly between current and
// neighbor average.
inline WeightMatrix
smooth_weights(const WeightMatrix &W, const Adjacency &adj,
                std::size_t nu, double omega) {
	WeightMatrix Wcur = W;
	const std::size_t n = W.size();
	const std::size_t nh = (n > 0) ? W[0].size() : 0;
	for (std::size_t pass = 0; pass < nu; ++pass) {
		WeightMatrix Wnext(n, std::vector<double>(nh, 0.0));
		for (std::size_t v = 0; v < n; ++v) {
			const auto &nbrs = adj[v];
			const std::size_t count = nbrs.size();
			std::vector<double> row(nh, 0.0);
			for (std::size_t i = 0; i < nh; ++i) {
				double nb_avg = 0.0;
				if (count > 0) {
					for (int u : nbrs) nb_avg += Wcur[u][i];
					nb_avg /= static_cast<double>(count);
				}
				row[i] = (1.0 - omega) * Wcur[v][i] + omega * nb_avg;
			}
			Wnext[v] = std::move(row);
		}
		Wcur = std::move(Wnext);
	}
	return Wcur;
}

// Top-K sparsification with renormalization. Per row, keep the K
// entries with largest absolute weight, zero the rest, then divide
// remaining entries by their sum so each row sums to 1.
//
// Output: per row, a vector of (handle_idx, normalized_weight) pairs
// of length min(K, row.size()).
inline std::vector<std::vector<std::pair<int, double>>>
sparsify_top_k(const WeightMatrix &W, std::size_t K) {
	const std::size_t n = W.size();
	std::vector<std::vector<std::pair<int, double>>> out(n);
	for (std::size_t v = 0; v < n; ++v) {
		const auto &row = W[v];
		// Collect (idx, w) pairs.
		std::vector<std::pair<int, double>> pairs;
		pairs.reserve(row.size());
		for (std::size_t i = 0; i < row.size(); ++i) {
			pairs.emplace_back(static_cast<int>(i), row[i]);
		}
		// Selection sort the top `take` entries by |w|. Small K, simple
		// is fine — beats partial-sort overhead at K <= 8.
		const std::size_t take = (K < pairs.size()) ? K : pairs.size();
		for (std::size_t k = 0; k < take; ++k) {
			std::size_t best = k;
			for (std::size_t j = k + 1; j < pairs.size(); ++j) {
				if (std::fabs(pairs[j].second) > std::fabs(pairs[best].second)) {
					best = j;
				}
			}
			std::swap(pairs[k], pairs[best]);
		}
		// Truncate + renormalize.
		std::vector<std::pair<int, double>> kept;
		kept.reserve(take);
		double sum = 0.0;
		for (std::size_t k = 0; k < take; ++k) {
			kept.push_back(pairs[k]);
			sum += pairs[k].second;
		}
		if (sum != 0.0) {
			for (auto &p : kept) p.second /= sum;
		}
		out[v] = std::move(kept);
	}
	return out;
}

// Sum of the kept weights per vertex (for partition-of-unity check on
// the sparse representation).
inline double sparse_row_sum(const std::vector<std::pair<int, double>> &row) {
	double s = 0.0;
	for (const auto &p : row) s += p.second;
	return s;
}

} // namespace direct_delta_mush
} // namespace curvenet

#endif
