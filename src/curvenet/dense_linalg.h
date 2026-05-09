// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
#ifndef CURVENET_DENSE_LINALG_H
#define CURVENET_DENSE_LINALG_H

// C++ mirror of `lean/Curvenet/DenseLinAlg.lean`. Dense matrix
// primitives + Gaussian elimination with partial pivoting. The
// production runtime swaps `solve` for an Eigen sparse SimplicialLLT
// once the cut-mesh sizes warrant it; this naive implementation keeps
// the pipeline self-contained for tests and small instances.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace curvenet {
namespace dense {

inline std::vector<double> zeros(std::size_t n, std::size_t m) {
	return std::vector<double>(n * m, 0.0);
}

inline std::vector<double> identity(std::size_t n) {
	std::vector<double> a(n * n, 0.0);
	for (std::size_t i = 0; i < n; ++i) {
		a[i * n + i] = 1.0;
	}
	return a;
}

inline double get_at(const std::vector<double> &a, std::size_t m, std::size_t i, std::size_t j) {
	return a[i * m + j];
}
inline void   set_at(std::vector<double> &a, std::size_t m, std::size_t i, std::size_t j, double v) {
	a[i * m + j] = v;
}

inline std::vector<double> mat_mul(std::size_t n, std::size_t k, std::size_t m,
                                    const std::vector<double> &a,
                                    const std::vector<double> &b) {
	std::vector<double> out(n * m, 0.0);
	for (std::size_t i = 0; i < n; ++i) {
		for (std::size_t j = 0; j < m; ++j) {
			double s = 0.0;
			for (std::size_t p = 0; p < k; ++p) {
				s += get_at(a, k, i, p) * get_at(b, m, p, j);
			}
			set_at(out, m, i, j, s);
		}
	}
	return out;
}

inline std::vector<double> transpose(std::size_t n, std::size_t m, const std::vector<double> &a) {
	std::vector<double> out(m * n, 0.0);
	for (std::size_t i = 0; i < n; ++i) {
		for (std::size_t j = 0; j < m; ++j) {
			set_at(out, n, j, i, get_at(a, m, i, j));
		}
	}
	return out;
}

inline std::vector<double> mat_vec(std::size_t n, std::size_t m,
                                    const std::vector<double> &a,
                                    const std::vector<double> &v) {
	std::vector<double> out(n, 0.0);
	for (std::size_t i = 0; i < n; ++i) {
		double s = 0.0;
		for (std::size_t j = 0; j < m; ++j) {
			s += get_at(a, m, i, j) * v[j];
		}
		out[i] = s;
	}
	return out;
}

// Solve A · x = b for n×n A and n-vector b via Gaussian elimination
// with partial pivoting. Singular systems silently return zeros for
// the affected rows (matching the Lean fallback).
inline std::vector<double> solve(std::size_t n, const std::vector<double> &A_in,
                                  const std::vector<double> &b) {
	const std::size_t W = n + 1;
	std::vector<double> aug(n * W, 0.0);
	for (std::size_t i = 0; i < n; ++i) {
		for (std::size_t j = 0; j < n; ++j) {
			aug[i * W + j] = A_in[i * n + j];
		}
		aug[i * W + n] = b[i];
	}
	for (std::size_t k = 0; k < n; ++k) {
		std::size_t piv = k;
		double best = std::abs(aug[k * W + k]);
		for (std::size_t r = k + 1; r < n; ++r) {
			const double v = std::abs(aug[r * W + k]);
			if (v > best) {
				best = v;
				piv = r;
			}
		}
		if (piv != k) {
			for (std::size_t j = 0; j < W; ++j) {
				std::swap(aug[k * W + j], aug[piv * W + j]);
			}
		}
		const double pivot = aug[k * W + k];
		if (pivot == 0.0) {
			continue;
		}
		for (std::size_t r = k + 1; r < n; ++r) {
			const double factor = aug[r * W + k] / pivot;
			for (std::size_t j = k; j < W; ++j) {
				aug[r * W + j] -= factor * aug[k * W + j];
			}
		}
	}
	std::vector<double> x(n, 0.0);
	for (std::size_t i = 0; i < n; ++i) {
		const std::size_t row = n - 1 - i;
		double s = aug[row * W + n];
		for (std::size_t j = row + 1; j < n; ++j) {
			s -= aug[row * W + j] * x[j];
		}
		const double pivot = aug[row * W + row];
		x[row] = (pivot == 0.0) ? 0.0 : (s / pivot);
	}
	return x;
}

inline std::vector<double> solve_multi(std::size_t n, std::size_t k,
                                        const std::vector<double> &A,
                                        const std::vector<double> &B) {
	std::vector<double> X(n * k, 0.0);
	for (std::size_t col = 0; col < k; ++col) {
		std::vector<double> b_col(n, 0.0);
		for (std::size_t i = 0; i < n; ++i) {
			b_col[i] = B[i * k + col];
		}
		const std::vector<double> x_col = solve(n, A, b_col);
		for (std::size_t i = 0; i < n; ++i) {
			X[i * k + col] = x_col[i];
		}
	}
	return X;
}

inline bool vec_within_eps(const std::vector<double> &a, const std::vector<double> &b, double eps) {
	if (a.size() != b.size()) {
		return false;
	}
	for (std::size_t i = 0; i < a.size(); ++i) {
		if (std::abs(a[i] - b[i]) >= eps) {
			return false;
		}
	}
	return true;
}

inline bool mat_within_eps(const std::vector<double> &a, const std::vector<double> &b,
                            std::size_t n, std::size_t m, double eps) {
	for (std::size_t i = 0; i < n; ++i) {
		for (std::size_t j = 0; j < m; ++j) {
			if (std::abs(get_at(a, m, i, j) - get_at(b, m, i, j)) >= eps) {
				return false;
			}
		}
	}
	return true;
}

} // namespace dense
} // namespace curvenet

#endif
