// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
//
// Mirrors `lean/Curvenet/DenseLinAlg.lean`'s instance theorems.
#include "test_helpers.h"

#include "curvenet/dense_linalg.h"

#include <cmath>
#include <vector>

namespace dl = curvenet::dense;

int main() {
	bool ok = true;

	ok &= rc::check("identity solves trivially", [] {
		const std::vector<double> A = dl::identity(3);
		const std::vector<double> b = { 1.0, 2.0, 3.0 };
		const std::vector<double> x = dl::solve(3, A, b);
		RC_ASSERT(dl::vec_within_eps(x, b, 1e-12));
	});

	ok &= rc::check("diagonal solve diag(2,3,4)·x = (4,9,16) -> x = (2,3,4)", [] {
		const std::vector<double> A = { 2.0, 0.0, 0.0,
		                                 0.0, 3.0, 0.0,
		                                 0.0, 0.0, 4.0 };
		const std::vector<double> b = { 4.0, 9.0, 16.0 };
		const std::vector<double> x = dl::solve(3, A, b);
		const std::vector<double> e = { 2.0, 3.0, 4.0 };
		RC_ASSERT(dl::vec_within_eps(x, e, 1e-12));
	});

	ok &= rc::check("2x2 SPD solve", [] {
		const std::vector<double> A = { 2.0, 1.0, 1.0, 2.0 };
		const std::vector<double> b = { 3.0, 3.0 };
		const std::vector<double> x = dl::solve(2, A, b);
		const std::vector<double> e = { 1.0, 1.0 };
		RC_ASSERT(dl::vec_within_eps(x, e, 1e-12));
	});

	ok &= rc::check("pivoting required (top row leading zero)", [] {
		const std::vector<double> A = { 0.0, 1.0, 1.0, 0.0 };
		const std::vector<double> b = { 2.0, 3.0 };
		const std::vector<double> x = dl::solve(2, A, b);
		const std::vector<double> e = { 3.0, 2.0 };
		RC_ASSERT(dl::vec_within_eps(x, e, 1e-12));
	});

	ok &= rc::check("solve round-trip A·x reproduces b", [] {
		const std::vector<double> A = { 2.0, 0.0, 1.0,
		                                 0.0, 3.0, 0.0,
		                                 1.0, 0.0, 4.0 };
		const std::vector<double> b = { 5.0, 6.0, 9.0 };
		const std::vector<double> x = dl::solve(3, A, b);
		const std::vector<double> bp = dl::mat_vec(3, 3, A, x);
		RC_ASSERT(dl::vec_within_eps(bp, b, 1e-10));
	});

	ok &= rc::check("(A^T A)^T = A^T A (symmetry of normal equation matrix)", [] {
		const std::vector<double> A = { 1.0, 2.0, 3.0,
		                                 4.0, 5.0, 6.0,
		                                 7.0, 8.0, 0.0 };
		const std::vector<double> At  = dl::transpose(3, 3, A);
		const std::vector<double> AtA = dl::mat_mul(3, 3, 3, At, A);
		const std::vector<double> AtAt = dl::transpose(3, 3, AtA);
		RC_ASSERT(dl::mat_within_eps(AtA, AtAt, 3, 3, 1e-12));
	});

	ok &= rc::check("solve_multi: 3 RHS columns", [] {
		const std::vector<double> A = dl::identity(3);
		const std::vector<double> B = { 1.0, 2.0, 3.0,
		                                 4.0, 5.0, 6.0,
		                                 7.0, 8.0, 9.0 };
		const std::vector<double> X = dl::solve_multi(3, 3, A, B);
		// I · B = B
		RC_ASSERT(dl::mat_within_eps(X, B, 3, 3, 1e-12));
	});

	ok &= rc::check("Cholesky: L · Lᵀ recovers A on 2x2 SPD", [] {
		const std::vector<double> A = { 4.0, 2.0,
		                                  2.0, 3.0 };
		const std::vector<double> L  = dl::cholesky_factor(2, A);
		const std::vector<double> LT = dl::transpose(2, 2, L);
		const std::vector<double> LLT = dl::mat_mul(2, 2, 2, L, LT);
		RC_ASSERT(dl::mat_within_eps(LLT, A, 2, 2, 1e-12));
	});

	ok &= rc::check("Cholesky: solve_with_cholesky on 2x2 SPD", [] {
		const std::vector<double> A = { 4.0, 2.0,
		                                  2.0, 3.0 };
		const std::vector<double> b = { 10.0, 8.0 };
		const std::vector<double> L = dl::cholesky_factor(2, A);
		const std::vector<double> x = dl::solve_with_cholesky(2, L, b);
		const std::vector<double> expected = { 1.75, 1.5 };
		RC_ASSERT(dl::vec_within_eps(x, expected, 1e-12));
	});

	ok &= rc::check("Cholesky agrees with LU on 4x4 1D Laplacian", [] {
		const std::vector<double> A = {
			 2.0, -1.0,  0.0,  0.0,
			-1.0,  2.0, -1.0,  0.0,
			 0.0, -1.0,  2.0, -1.0,
			 0.0,  0.0, -1.0,  2.0
		};
		const std::vector<double> b = { 1.0, 0.0, 0.0, 0.0 };
		const std::vector<double> xLU = dl::solve(4, A, b);
		const std::vector<double> L   = dl::cholesky_factor(4, A);
		const std::vector<double> xCh = dl::solve_with_cholesky(4, L, b);
		RC_ASSERT(dl::vec_within_eps(xLU, xCh, 1e-10));
	});

	ok &= rc::check("Cholesky factor reused across multiple RHS", [] {
		const std::vector<double> A = { 4.0, 2.0,
		                                  2.0, 3.0 };
		const std::vector<double> L = dl::cholesky_factor(2, A);
		// Two RHS columns with known answers.
		const std::vector<double> x1 =
			dl::solve_with_cholesky(2, L, { 10.0, 8.0 });
		const std::vector<double> x2 =
			dl::solve_with_cholesky(2, L, { 4.0, 3.0 });
		RC_ASSERT(dl::vec_within_eps(x1, { 1.75, 1.5 }, 1e-12));
		RC_ASSERT(dl::vec_within_eps(x2, { 0.75, 0.5 }, 1e-12));
	});

	return ok ? 0 : 1;
}
