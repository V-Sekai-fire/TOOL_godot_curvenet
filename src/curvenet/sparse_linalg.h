// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
#ifndef CURVENET_SPARSE_LINALG_H
#define CURVENET_SPARSE_LINALG_H

// C++ mirror of `lean/Curvenet/SparseLinAlg.lean`. CSR sparse matrix +
// preconditioned conjugate gradient. The dense factorization in
// `dense_linalg.h` works fine for the demo plane (nv ≈ 100) but blows
// up at character-mesh scale: dense LU on a 100k × 100k matrix is
// 80 GB and ~10 hours per factor. CSR + CG keeps memory O(nnz) ≈ O(n)
// and runs in O(nnz · √κ) per RHS, scaling gracefully past 100k verts.

#include <cmath>
#include <cstddef>
#include <vector>

namespace curvenet {
namespace sparse {

// Compressed sparse row matrix. row_ptr.size() == rows + 1; row i's
// nonzeros sit in col_idx[row_ptr[i] .. row_ptr[i+1]) with the matching
// entries in values[].
struct SparseMatrixCSR {
	std::size_t            rows = 0;
	std::size_t            cols = 0;
	std::vector<int>       row_ptr;
	std::vector<int>       col_idx;
	std::vector<double>    values;
};

// y = A · x. O(nnz(A)).
inline std::vector<double> spmv(const SparseMatrixCSR &A,
                                  const std::vector<double> &x) {
	std::vector<double> y(A.rows, 0.0);
	for (std::size_t i = 0; i < A.rows; ++i) {
		const int rs = A.row_ptr[i];
		const int re = A.row_ptr[i + 1];
		double s = 0.0;
		for (int p = rs; p < re; ++p) {
			s += A.values[p] * x[A.col_idx[p]];
		}
		y[i] = s;
	}
	return y;
}

// Multi-RHS sparse mat-vec: Y[i, c] = Σ A[i, j] · X[j, c] for k columns.
// Out is rows × k row-major. O(nnz(A) · k).
inline std::vector<double> spmv_multi(const SparseMatrixCSR &A,
                                        const std::vector<double> &X,
                                        std::size_t k) {
	std::vector<double> Y(A.rows * k, 0.0);
	for (std::size_t i = 0; i < A.rows; ++i) {
		const int rs = A.row_ptr[i];
		const int re = A.row_ptr[i + 1];
		for (int p = rs; p < re; ++p) {
			const double aij = A.values[p];
			const std::size_t j = static_cast<std::size_t>(A.col_idx[p]);
			for (std::size_t c = 0; c < k; ++c) {
				Y[i * k + c] += aij * X[j * k + c];
			}
		}
	}
	return Y;
}

inline double dot(const std::vector<double> &a, const std::vector<double> &b) {
	double s = 0.0;
	for (std::size_t i = 0; i < a.size(); ++i) {
		s += a[i] * b[i];
	}
	return s;
}

// y ← y + α·x  (in place).
inline void axpy_inplace(double alpha,
                          const std::vector<double> &x,
                          std::vector<double> &y) {
	for (std::size_t i = 0; i < x.size(); ++i) {
		y[i] += alpha * x[i];
	}
}

// out ← α·x + β·y
inline std::vector<double> saxpby(double alpha, const std::vector<double> &x,
                                    double beta,  const std::vector<double> &y) {
	std::vector<double> out(y.size(), 0.0);
	for (std::size_t i = 0; i < y.size(); ++i) {
		out[i] = alpha * x[i] + beta * y[i];
	}
	return out;
}

// diag(A) — used as the Jacobi preconditioner M⁻¹ = diag(1/A_ii).
inline std::vector<double> diagonal(const SparseMatrixCSR &A) {
	std::vector<double> d(A.rows, 0.0);
	for (std::size_t i = 0; i < A.rows; ++i) {
		const int rs = A.row_ptr[i];
		const int re = A.row_ptr[i + 1];
		for (int p = rs; p < re; ++p) {
			if (static_cast<std::size_t>(A.col_idx[p]) == i) {
				d[i] = A.values[p];
				break;
			}
		}
	}
	return d;
}

// y_i = b_i / d_i; zero diagonal entries map to 0 (consistent with the
// dense solver's promoted-vertex slot convention — caller overlays
// the constraint value back).
inline std::vector<double> apply_jacobi(const std::vector<double> &d,
                                          const std::vector<double> &b) {
	std::vector<double> y(b.size(), 0.0);
	for (std::size_t i = 0; i < b.size(); ++i) {
		const double dii = d[i];
		y[i] = (dii == 0.0) ? 0.0 : (b[i] / dii);
	}
	return y;
}

// Preconditioned conjugate gradient. SPD only. Returns x ≈ A⁻¹·b after
// either `max_iter` iterations or once `‖r‖² < tol²`.
inline std::vector<double> cg(const SparseMatrixCSR &A,
                                const std::vector<double> &b,
                                std::size_t max_iter,
                                double tol) {
	const std::size_t n = A.rows;
	const std::vector<double> d = diagonal(A);
	std::vector<double> x(n, 0.0);
	std::vector<double> r = b;
	std::vector<double> z = apply_jacobi(d, r);
	std::vector<double> p = z;
	double rz_old = dot(r, z);
	const double tol_sq = tol * tol;
	for (std::size_t iter = 0; iter < max_iter; ++iter) {
		const std::vector<double> Ap = spmv(A, p);
		const double pAp = dot(p, Ap);
		if (pAp == 0.0) {
			break;
		}
		const double alpha = rz_old / pAp;
		axpy_inplace(alpha, p, x);
		axpy_inplace(-alpha, Ap, r);
		const double rr = dot(r, r);
		if (rr < tol_sq) {
			break;
		}
		z = apply_jacobi(d, r);
		const double rz_new = dot(r, z);
		const double beta = (rz_old == 0.0) ? 0.0 : (rz_new / rz_old);
		p = saxpby(1.0, z, beta, p);
		rz_old = rz_new;
	}
	return x;
}

// CG with an explicit initial guess `x0`. Identical fixed point to
// `cg` (same A, same b → same true solution); converges faster when
// `x0` is close. The runtime warm-starts each frame's solve with the
// previous frame's iterate, so on smooth handle drags the residual
// drops below `tol` after a handful of iterations instead of restarting
// from zero.
inline std::vector<double> cg_with_guess(const SparseMatrixCSR &A,
                                          const std::vector<double> &b,
                                          const std::vector<double> &x0,
                                          std::size_t max_iter,
                                          double tol) {
	const std::size_t n = A.rows;
	const std::vector<double> d = diagonal(A);
	std::vector<double> x = x0;
	const std::vector<double> Ax0 = spmv(A, x0);
	std::vector<double> r = saxpby(1.0, b, -1.0, Ax0);
	std::vector<double> z = apply_jacobi(d, r);
	std::vector<double> p = z;
	double rz_old = dot(r, z);
	const double tol_sq = tol * tol;
	for (std::size_t iter = 0; iter < max_iter; ++iter) {
		const std::vector<double> Ap = spmv(A, p);
		const double pAp = dot(p, Ap);
		if (pAp == 0.0) {
			break;
		}
		const double alpha = rz_old / pAp;
		axpy_inplace(alpha, p, x);
		axpy_inplace(-alpha, Ap, r);
		const double rr = dot(r, r);
		if (rr < tol_sq) {
			break;
		}
		z = apply_jacobi(d, r);
		const double rz_new = dot(r, z);
		const double beta = (rz_old == 0.0) ? 0.0 : (rz_new / rz_old);
		p = saxpby(1.0, z, beta, p);
		rz_old = rz_new;
	}
	(void)n;
	return x;
}

// Multi-RHS CG: independent solve per column. Each call shares the
// sparse representation but does its own iteration.
inline std::vector<double> cg_multi(const SparseMatrixCSR &A,
                                      const std::vector<double> &B,
                                      std::size_t k,
                                      std::size_t max_iter,
                                      double tol) {
	const std::size_t n = A.rows;
	std::vector<double> X(n * k, 0.0);
	std::vector<double> b_col(n, 0.0);
	for (std::size_t col = 0; col < k; ++col) {
		for (std::size_t i = 0; i < n; ++i) {
			b_col[i] = B[i * k + col];
		}
		const std::vector<double> x_col = cg(A, b_col, max_iter, tol);
		for (std::size_t i = 0; i < n; ++i) {
			X[i * k + col] = x_col[i];
		}
	}
	return X;
}

} // namespace sparse
} // namespace curvenet

#endif
