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

// LU factorization with partial pivoting. Self-contained replacement for
// Eigen's `SimplicialLLT` — same factor-once / solve-many semantics, no
// new thirdparty dependency. For symmetric positive-definite inputs this
// is asymptotically twice the factorization cost of Cholesky but the
// per-RHS solve cost is identical (both `O(n²)`), and the factorization
// only runs once at bind time.
struct LUFactor {
	std::size_t        n     = 0;
	std::vector<double> a;          // n×n; lower triangular (unit diagonal,
	                                  // implicit) below the diagonal,
	                                  // upper triangular on/above
	std::vector<int>   piv;          // row permutation from partial pivoting
	bool               valid = false;
};

inline LUFactor factorize_lu(std::size_t n, const std::vector<double> &A_in) {
	LUFactor f;
	f.n = n;
	f.a = A_in;
	f.piv.resize(n);
	for (std::size_t i = 0; i < n; ++i) {
		f.piv[i] = static_cast<int>(i);
	}
	for (std::size_t k = 0; k < n; ++k) {
		std::size_t pivot_row = k;
		double best = std::abs(f.a[k * n + k]);
		for (std::size_t r = k + 1; r < n; ++r) {
			const double v = std::abs(f.a[r * n + k]);
			if (v > best) {
				best = v;
				pivot_row = r;
			}
		}
		if (pivot_row != k) {
			for (std::size_t j = 0; j < n; ++j) {
				std::swap(f.a[k * n + j], f.a[pivot_row * n + j]);
			}
			std::swap(f.piv[k], f.piv[pivot_row]);
		}
		const double pivot = f.a[k * n + k];
		if (pivot == 0.0) {
			continue; // singular column; zero row in LHS (e.g. promoted-
			          // vertex slot). Solve will return 0 here.
		}
		for (std::size_t r = k + 1; r < n; ++r) {
			const double factor = f.a[r * n + k] / pivot;
			f.a[r * n + k] = factor;
			for (std::size_t j = k + 1; j < n; ++j) {
				f.a[r * n + j] -= factor * f.a[k * n + j];
			}
		}
	}
	f.valid = true;
	return f;
}

// Solve A·x = b given a previously-computed LU factorization. O(n²).
inline std::vector<double> solve_with_lu(const LUFactor &f,
                                          const std::vector<double> &b) {
	const std::size_t n = f.n;
	std::vector<double> x(n, 0.0);
	if (!f.valid || b.size() != n) {
		return x;
	}
	// Apply row permutation: y = P b
	std::vector<double> y(n, 0.0);
	for (std::size_t i = 0; i < n; ++i) {
		y[i] = b[f.piv[i]];
	}
	// Forward substitution: L y = (Pb), L unit lower triangular.
	for (std::size_t i = 0; i < n; ++i) {
		double sum = y[i];
		for (std::size_t j = 0; j < i; ++j) {
			sum -= f.a[i * n + j] * y[j];
		}
		y[i] = sum;
	}
	// Back substitution: U x = y.
	for (std::size_t i = 0; i < n; ++i) {
		const std::size_t row = n - 1 - i;
		double sum = y[row];
		for (std::size_t j = row + 1; j < n; ++j) {
			sum -= f.a[row * n + j] * x[j];
		}
		const double pivot = f.a[row * n + row];
		x[row] = (pivot == 0.0) ? 0.0 : sum / pivot;
	}
	return x;
}

// Solve A·X = B for k RHS columns using a precomputed LU factor. Each
// column is back-substituted independently, sharing the cached factor.
inline std::vector<double> solve_multi_with_lu(const LUFactor &f, std::size_t k,
                                                  const std::vector<double> &B) {
	const std::size_t n = f.n;
	std::vector<double> X(n * k, 0.0);
	std::vector<double> b_col(n, 0.0);
	for (std::size_t col = 0; col < k; ++col) {
		for (std::size_t i = 0; i < n; ++i) {
			b_col[i] = B[i * k + col];
		}
		const std::vector<double> x_col = solve_with_lu(f, b_col);
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

// ============================================================
// Cholesky factorization for SPD matrices
// ============================================================
//
// Mirrors `Curvenet.DenseLinAlg.choleskyFactor` / `forwardSolve` /
// `backwardSolve` / `solveWithCholesky` in the Lean spec. Used by the
// meshlet Schur-complement path: each per-meshlet local matrix is
// SPD after symmetric Dirichlet pinning; we factor each one once at
// bind time and reuse the factor for fast back-substitution per
// frame.

inline std::vector<double> cholesky_factor(std::size_t n,
                                              const std::vector<double> &A) {
	std::vector<double> L(n * n, 0.0);
	for (std::size_t j = 0; j < n; ++j) {
		double s = 0.0;
		for (std::size_t k = 0; k < j; ++k) {
			const double v = L[j * n + k];
			s += v * v;
		}
		const double diag_sq = A[j * n + j] - s;
		const double diag    = std::sqrt(diag_sq);
		L[j * n + j] = diag;
		for (std::size_t i = j + 1; i < n; ++i) {
			double t = 0.0;
			for (std::size_t k = 0; k < j; ++k) {
				t += L[i * n + k] * L[j * n + k];
			}
			L[i * n + j] = (A[i * n + j] - t) / diag;
		}
	}
	return L;
}

inline std::vector<double> forward_solve(std::size_t n,
                                            const std::vector<double> &L,
                                            const std::vector<double> &b) {
	std::vector<double> y(n, 0.0);
	for (std::size_t i = 0; i < n; ++i) {
		double s = 0.0;
		for (std::size_t j = 0; j < i; ++j) {
			s += L[i * n + j] * y[j];
		}
		y[i] = (b[i] - s) / L[i * n + i];
	}
	return y;
}

inline std::vector<double> backward_solve(std::size_t n,
                                              const std::vector<double> &L,
                                              const std::vector<double> &y) {
	std::vector<double> x(n, 0.0);
	for (std::size_t ii = 0; ii < n; ++ii) {
		const std::size_t i = n - 1 - ii;
		double s = 0.0;
		for (std::size_t j = i + 1; j < n; ++j) {
			s += L[j * n + i] * x[j];   // Lᵀ[i, j] = L[j, i]
		}
		x[i] = (y[i] - s) / L[i * n + i];
	}
	return x;
}

inline std::vector<double> solve_with_cholesky(std::size_t n,
                                                   const std::vector<double> &L,
                                                   const std::vector<double> &b) {
	return backward_solve(n, L, forward_solve(n, L, b));
}

} // namespace dense
} // namespace curvenet

#endif
