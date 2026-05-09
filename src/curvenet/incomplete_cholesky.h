// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
//
// C++ mirror of `lean/Curvenet/IncompleteCholesky.lean`. ICC(0) =
// no-fill incomplete Cholesky: factorise A as L · L^T where L
// keeps exactly the sparsity pattern of tril(A). Backsolve via
// forward-then-backward triangular solve.
//
// Used by `sparse_linalg::cg_icc` (loop 100/3) as a preconditioner
// for the 81k cut-mesh Laplacian. Loops 8 — 100/2 ruled out three
// preconditioner families (multilevel Schwarz, HEM, kernel proj.)
// because the omega·D^{-1}·r Jacobi smoother breaks on the matrix's
// 7-decade diagonal spread. ICC(0) doesn't smooth — it directly
// approximates A^{-1} via L^{-T}·L^{-1} so the preconditioner is
// independent of the diagonal spread.
//
// Cost:
//   factor    O(nnz(A)) once at bind time (per matrix)
//   backsolve 2 · O(nnz(A)) per CG iter (forward + backward)
// Backsolve is the same cost as one A·x mat-vec, so each ICC-PCG
// iter is ~3× a plain CG iter (mat-vec + 2 backsolves vs. mat-vec).
// Standard result: kappa(M^{-1}A) ~ sqrt(kappa(A)) for many SPD
// problems, so iter count drops from ~sqrt(kappa) to ~kappa^{1/4}.

#ifndef CURVENET_INCOMPLETE_CHOLESKY_H
#define CURVENET_INCOMPLETE_CHOLESKY_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <unordered_map>
#include <vector>

#include "sparse_linalg.h"

namespace curvenet {
namespace incomplete_cholesky {

// Lower-triangular CSR factor `L` such that L · L^T ≈ A.
// `L.row_ptr[i]..L.row_ptr[i+1]` holds entries `L[i, j]` with j ≤ i.
struct IncompleteCholeskyFactor {
    sparse::SparseMatrixCSR L;
    bool                    breakdown;  // true if any diag(L)^2 ≤ 0
};

// Factorise an SPD CSR matrix A into ICC(0) form. The output L
// stores only the lower-triangular non-zeros of A; entries that
// would fill in are skipped (no-fill variant). Diagonal entries
// of L are sqrt(...) of a residual, must be > 0 for breakdown=false.
//
// `diag_shift` adds `alpha * |A[i,i]|` to each diagonal entry before
// factorisation (Manteuffel 1980 shifted-ICC). Use 0.0 for plain
// ICC(0); bump up if the no-fill variant breaks down. The shift
// makes the factorisation an approximation of `A + alpha · diag(A)`
// rather than `A`, which CG converges to the same solution of `A x = b`
// because PCG only requires M to be SPD and a "good approximation"
// — a small shift tightens kappa(M^{-1}A) modestly while ruling out
// breakdown. Empirically `alpha = 1e-3` clears most cot-Laplacians.
inline IncompleteCholeskyFactor factor(const sparse::SparseMatrixCSR &A,
                            double diag_shift = 0.0) {
    const std::size_t n = A.rows;
    IncompleteCholeskyFactor out;
    out.breakdown = false;

    // Build L's sparsity pattern from tril(A): for each row i,
    // collect column indices j ≤ i where A[i, j] is non-zero.
    std::vector<std::vector<int>>    cols(n);
    std::vector<std::vector<double>> vals(n);
    for (std::size_t i = 0; i < n; ++i) {
        const int rs = A.row_ptr[i];
        const int re = A.row_ptr[i + 1];
        for (int k = rs; k < re; ++k) {
            const int j = A.col_idx[k];
            if (j > static_cast<int>(i)) continue;
            cols[i].push_back(j);
            double v = A.values[k];
            if (j == static_cast<int>(i) && diag_shift != 0.0) {
                v += diag_shift * std::fabs(v);
            }
            vals[i].push_back(v);
        }
        // Sort by column for deterministic row layout + fast lookup.
        std::vector<std::size_t> perm(cols[i].size());
        for (std::size_t k = 0; k < perm.size(); ++k) perm[k] = k;
        std::sort(perm.begin(), perm.end(),
                     [&](std::size_t a, std::size_t b) {
                         return cols[i][a] < cols[i][b];
                     });
        std::vector<int>    cs(cols[i].size());
        std::vector<double> vs(vals[i].size());
        for (std::size_t k = 0; k < perm.size(); ++k) {
            cs[k] = cols[i][perm[k]];
            vs[k] = vals[i][perm[k]];
        }
        cols[i] = std::move(cs);
        vals[i] = std::move(vs);
    }

    // Per-row column→index map for O(1) lookup of L[i, k] during the
    // inner sum over k. Built lazily as rows are processed.
    std::vector<std::unordered_map<int, int>> idx_map(n);
    for (std::size_t i = 0; i < n; ++i) {
        idx_map[i].reserve(cols[i].size() * 2);
        for (std::size_t k = 0; k < cols[i].size(); ++k) {
            idx_map[i][cols[i][k]] = static_cast<int>(k);
        }
    }

    // Crout-style column elimination: process column j in order,
    // computing L[j, j] then L[i, j] for i > j with A[i, j] non-zero.
    for (std::size_t j = 0; j < n; ++j) {
        // L[j, j] = sqrt(A[j, j] - sum_{k<j} L[j, k]^2).
        double s = 0.0;
        bool found_diag = false;
        std::size_t diag_pos = 0;
        for (std::size_t k = 0; k < cols[j].size(); ++k) {
            const int c = cols[j][k];
            if (c == static_cast<int>(j)) {
                found_diag = true;
                diag_pos = k;
                s += vals[j][k];          // start with A[j, j]
                continue;
            }
            // Else c < j; subtract L[j, c]^2.
            const double ljc = vals[j][k];
            s -= ljc * ljc;
        }
        if (!found_diag || s <= 0.0) {
            out.breakdown = true;
            return out;
        }
        const double ljj = std::sqrt(s);
        vals[j][diag_pos] = ljj;

        // For each i > j with A[i, j] in pattern: update L[i, j].
        // We scan rows i where j ∈ cols[i]; for the no-fill variant
        // we only need i in the pattern of column j, which is the
        // pattern of A's column j (= rows with non-zero in column j).
        // Build column-j lookup once (already in idx_map[i]).
        for (std::size_t i = j + 1; i < n; ++i) {
            const auto it = idx_map[i].find(static_cast<int>(j));
            if (it == idx_map[i].end()) continue;
            const int pos_ij = it->second;

            // Sum over k < j where both L[i, k] and L[j, k] exist.
            double t = vals[i][pos_ij];   // start with A[i, j]
            for (std::size_t kk = 0; kk < cols[j].size(); ++kk) {
                const int k = cols[j][kk];
                if (k >= static_cast<int>(j)) break;
                const auto itk = idx_map[i].find(k);
                if (itk == idx_map[i].end()) continue;
                t -= vals[i][itk->second] * vals[j][kk];
            }
            vals[i][pos_ij] = t / ljj;
        }
    }

    // Pack into CSR.
    out.L.rows = n;
    out.L.cols = n;
    out.L.row_ptr.assign(n + 1, 0);
    for (std::size_t i = 0; i < n; ++i) out.L.row_ptr[i + 1] = out.L.row_ptr[i] + static_cast<int>(cols[i].size());
    out.L.col_idx.resize(out.L.row_ptr[n]);
    out.L.values.resize(out.L.row_ptr[n]);
    for (std::size_t i = 0; i < n; ++i) {
        int p = out.L.row_ptr[i];
        for (std::size_t k = 0; k < cols[i].size(); ++k) {
            out.L.col_idx[p] = cols[i][k];
            out.L.values[p]  = vals[i][k];
            ++p;
        }
    }
    return out;
}

// Forward sub: solve L · y = b where L is lower-triangular CSR
// (entries strictly j ≤ i, diagonal stored last per row).
inline std::vector<double> forward_sub(const sparse::SparseMatrixCSR &L,
                                          const std::vector<double> &b) {
    const std::size_t n = L.rows;
    std::vector<double> y(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        double s = b[i];
        double diag = 1.0;
        for (int k = L.row_ptr[i]; k < L.row_ptr[i + 1]; ++k) {
            const int j = L.col_idx[k];
            if (j == static_cast<int>(i)) {
                diag = L.values[k];
            } else if (j < static_cast<int>(i)) {
                s -= L.values[k] * y[j];
            }
        }
        y[i] = s / diag;
    }
    return y;
}

// Backward sub: solve L^T · x = y. Iterates rows in reverse and
// accumulates contributions of already-resolved x[i+1..] into the
// current row's RHS via the column structure of L (= row structure
// of L^T).
inline std::vector<double> backward_sub(const sparse::SparseMatrixCSR &L,
                                            const std::vector<double> &y) {
    const std::size_t n = L.rows;
    std::vector<double> x(n, 0.0);
    std::vector<double> rhs = y;
    for (std::size_t jj = 0; jj < n; ++jj) {
        const std::size_t i = n - 1 - jj;
        double diag = 1.0;
        for (int k = L.row_ptr[i]; k < L.row_ptr[i + 1]; ++k) {
            if (L.col_idx[k] == static_cast<int>(i)) {
                diag = L.values[k];
                break;
            }
        }
        x[i] = rhs[i] / diag;
        // Subtract L[i, j] * x[i] from rhs[j] for all j < i.
        for (int k = L.row_ptr[i]; k < L.row_ptr[i + 1]; ++k) {
            const int j = L.col_idx[k];
            if (j < static_cast<int>(i)) rhs[j] -= L.values[k] * x[i];
        }
    }
    return x;
}

// Apply M^{-1} = L^{-T}·L^{-1} to a vector r.
//
// (Named `apply_minv` rather than `apply` to avoid ADL-collision
// with `std::apply` under C++17+ when this header is included from
// a TU that uses unqualified `apply`.)
inline std::vector<double> apply_minv(const IncompleteCholeskyFactor &fac,
                                          const std::vector<double> &r) {
    return backward_sub(fac.L, forward_sub(fac.L, r));
}

// Factor with progressive diagonal-shift retry. Tries shifts in
// {0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0} until the no-fill ICC(0)
// factorisation succeeds. Returns IncompleteCholeskyFactor{ breakdown=true } if
// none works. The shift used is recorded in `shift_used`.
inline IncompleteCholeskyFactor factor_with_retry(const sparse::SparseMatrixCSR &A,
                                       double *shift_used = nullptr) {
    IncompleteCholeskyFactor fac;
    for (double s : { 0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0 }) {
        fac = factor(A, s);
        if (shift_used) *shift_used = s;
        if (!fac.breakdown) return fac;
    }
    return fac;
}

// ICC(0)-PCG with an explicit initial guess. Mirror of
// `sparse::cg_with_guess` with the inner Jacobi preconditioner
// swapped for `apply(fac, ...)`. Same convergence semantics:
// when `x0` is close to the true solution, residual starts small
// and iter count drops.
inline std::vector<double> cg_icc_with_guess(
        const sparse::SparseMatrixCSR &A,
        const IncompleteCholeskyFactor &fac,
        const std::vector<double> &b,
        const std::vector<double> &x0,
        std::size_t max_iter,
        double tol) {
    std::vector<double> x = x0;
    const std::vector<double> Ax0 = sparse::spmv(A, x0);
    std::vector<double> r = sparse::saxpby(1.0, b, -1.0, Ax0);
    std::vector<double> z = apply_minv(fac, r);
    std::vector<double> p = z;
    double rz_old = sparse::dot(r, z);
    const double tol_sq = tol * tol;
    for (std::size_t iter = 0; iter < max_iter; ++iter) {
        const std::vector<double> Ap = sparse::spmv(A, p);
        const double pAp = sparse::dot(p, Ap);
        if (pAp == 0.0) break;
        const double alpha = rz_old / pAp;
        sparse::axpy_inplace(alpha, p, x);
        sparse::axpy_inplace(-alpha, Ap, r);
        if (sparse::dot(r, r) < tol_sq) break;
        z = apply_minv(fac, r);
        const double rz_new = sparse::dot(r, z);
        const double beta = (rz_old == 0.0) ? 0.0 : (rz_new / rz_old);
        p = sparse::saxpby(1.0, z, beta, p);
        rz_old = rz_new;
    }
    return x;
}

} // namespace incomplete_cholesky
} // namespace curvenet

#endif
