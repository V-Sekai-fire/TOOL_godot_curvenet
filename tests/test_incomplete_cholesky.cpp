// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
//
// RapidCheck mirror of IncompleteCholesky.lean.

#include "curvenet/incomplete_cholesky.h"
#include "curvenet/sparse_linalg.h"

#include <rapidcheck.h>
#include <cmath>
#include <vector>

namespace icc = curvenet::incomplete_cholesky;
namespace sp  = curvenet::sparse;

namespace {

// Build a CSR matrix from a dense row-major n×n array.
sp::SparseMatrixCSR dense_to_csr(const std::vector<double> &dense, std::size_t n) {
    sp::SparseMatrixCSR A;
    A.rows = n; A.cols = n;
    A.row_ptr.assign(n + 1, 0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            if (dense[i * n + j] != 0.0) ++A.row_ptr[i + 1];
        }
    }
    for (std::size_t i = 0; i < n; ++i) A.row_ptr[i + 1] += A.row_ptr[i];
    A.col_idx.resize(A.row_ptr[n]);
    A.values.resize(A.row_ptr[n]);
    std::vector<int> cur = A.row_ptr;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            const double v = dense[i * n + j];
            if (v != 0.0) {
                const int p = cur[i]++;
                A.col_idx[p] = static_cast<int>(j);
                A.values[p]  = v;
            }
        }
    }
    return A;
}

std::vector<double> csr_lt_dot_l(const sp::SparseMatrixCSR &L) {
    const std::size_t n = L.rows;
    std::vector<double> dense(n * n, 0.0);
    // Form dense L
    std::vector<double> Ldense(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (int k = L.row_ptr[i]; k < L.row_ptr[i + 1]; ++k) {
            Ldense[i * n + L.col_idx[k]] = L.values[k];
        }
    }
    // M = L · L^T
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (std::size_t k = 0; k < n; ++k) {
                s += Ldense[i * n + k] * Ldense[j * n + k];
            }
            dense[i * n + j] = s;
        }
    }
    return dense;
}

} // namespace

int main() {
    bool ok = true;

    // 3x3 tridiagonal SPD: diag 4, off-diag -1. Full pattern.
    const std::vector<double> Atri = {
         4.0, -1.0,  0.0,
        -1.0,  4.0, -1.0,
         0.0, -1.0,  4.0
    };
    const sp::SparseMatrixCSR A = dense_to_csr(Atri, 3);

    ok &= rc::check("ICC factor succeeds on tridiagonal SPD", [&] {
        const icc::IncompleteCholeskyFactor fac = icc::factor(A);
        RC_ASSERT(!fac.breakdown);
        RC_ASSERT(fac.L.rows == 3u);
    });

    ok &= rc::check("L is lower-triangular (no entries above diag)", [&] {
        const icc::IncompleteCholeskyFactor fac = icc::factor(A);
        for (std::size_t i = 0; i < fac.L.rows; ++i) {
            for (int k = fac.L.row_ptr[i]; k < fac.L.row_ptr[i + 1]; ++k) {
                RC_ASSERT(fac.L.col_idx[k] <= static_cast<int>(i));
            }
        }
    });

    ok &= rc::check("L · L^T == A on tridiagonal (full pattern preserved)", [&] {
        const icc::IncompleteCholeskyFactor fac = icc::factor(A);
        const auto LLT = csr_lt_dot_l(fac.L);
        for (std::size_t k = 0; k < 9; ++k) {
            RC_ASSERT(std::fabs(LLT[k] - Atri[k]) < 1e-12);
        }
    });

    ok &= rc::check("forward+back sub recovers x from b = A x", [&] {
        const icc::IncompleteCholeskyFactor fac = icc::factor(A);
        const std::vector<double> x_true = { 1.5, -2.0, 0.75 };
        const std::vector<double> b = sp::spmv(A, x_true);
        const std::vector<double> y = icc::forward_sub(fac.L, b);
        const std::vector<double> x = icc::backward_sub(fac.L, y);
        for (std::size_t i = 0; i < 3; ++i) {
            RC_ASSERT(std::fabs(x[i] - x_true[i]) < 1e-10);
        }
    });

    ok &= rc::check("ICC handles wide-diagonal SPD (6 decades)", [&] {
        const std::vector<double> Awide = {
            1e-2, 1e-3, 0.0,
            1e-3, 1.0,  1e-2,
            0.0,  1e-2, 1e+4
        };
        const sp::SparseMatrixCSR Aw = dense_to_csr(Awide, 3);
        const icc::IncompleteCholeskyFactor fac = icc::factor(Aw);
        RC_ASSERT(!fac.breakdown);
        const auto LLT = csr_lt_dot_l(fac.L);
        for (std::size_t k = 0; k < 9; ++k) {
            RC_ASSERT(std::fabs(LLT[k] - Awide[k]) < 1e-9);
        }
    });

    return ok ? 0 : 1;
}
