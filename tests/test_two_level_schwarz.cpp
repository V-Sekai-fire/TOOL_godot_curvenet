// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
//
// RapidCheck mirror of lean/Curvenet/TwoLevelSchwarz.lean's six
// native_decide proofs.

#include "curvenet/sparse_linalg.h"
#include "curvenet/two_level_schwarz.h"

#include <rapidcheck.h>
#include <cmath>
#include <vector>

namespace tls = curvenet::two_level_schwarz;
namespace sp  = curvenet::sparse;

namespace {

bool vec_close(const std::vector<double> &a, const std::vector<double> &b, double eps) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (std::fabs(a[i] - b[i]) >= eps) return false;
    }
    return true;
}

// Build CSR from a small dense matrix for the Galerkin tests.
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
    A.col_idx.resize(A.row_ptr.back());
    A.values.resize(A.row_ptr.back());
    std::vector<int> cursor = A.row_ptr;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            const double v = dense[i * n + j];
            if (v != 0.0) {
                const int idx = cursor[i]++;
                A.col_idx[idx] = static_cast<int>(j);
                A.values[idx]  = v;
            }
        }
    }
    return A;
}

double mat_at(const sp::SparseMatrixCSR &A, std::size_t i, std::size_t j) {
    const int rs = A.row_ptr[i];
    const int re = A.row_ptr[i + 1];
    for (int k = rs; k < re; ++k) {
        if (static_cast<std::size_t>(A.col_idx[k]) == j) return A.values[k];
    }
    return 0.0;
}

} // namespace

int main() {
    bool ok = true;
    const std::vector<int> cmap = { 0, 0, 1, 1 };

    ok &= rc::check("restrict sums fine values per cluster", [&] {
        const std::vector<double> f = { 1.0, 2.0, 3.0, 4.0 };
        const auto c = tls::restrict_fine(cmap, 2, f);
        RC_ASSERT(vec_close(c, { 3.0, 7.0 }, 1e-12));
    });

    ok &= rc::check("prolong distributes coarse to all members", [&] {
        const std::vector<double> c = { 5.0, 9.0 };
        const auto f = tls::prolong_coarse(cmap, 4, c);
        RC_ASSERT(vec_close(f, { 5.0, 5.0, 9.0, 9.0 }, 1e-12));
    });

    ok &= rc::check("restrict ∘ prolong scales by cluster size", [&] {
        const std::vector<double> c = { 5.0, 9.0 };
        const auto rt = tls::restrict_fine(cmap, 2, tls::prolong_coarse(cmap, 4, c));
        RC_ASSERT(vec_close(rt, { 10.0, 18.0 }, 1e-12));
    });

    ok &= rc::check("coarseSizes returns cluster member counts", [&] {
        const auto s = tls::coarse_sizes(cmap, 2);
        RC_ASSERT(s == std::vector<std::size_t>({ 2, 2 }));
    });

    ok &= rc::check("Galerkin on Dirichlet 1D Laplacian preserves symmetry", [&] {
        const std::vector<double> A = {
             2.0, -1.0,  0.0,  0.0,
            -1.0,  2.0, -1.0,  0.0,
             0.0, -1.0,  2.0, -1.0,
             0.0,  0.0, -1.0,  2.0,
        };
        const auto A_csr = dense_to_csr(A, 4);
        const auto Ac    = tls::galerkin_csr(cmap, 2, A_csr);
        // Symmetric: A_c[0,1] == A_c[1,0]
        RC_ASSERT(std::fabs(mat_at(Ac, 0, 1) - mat_at(Ac, 1, 0)) < 1e-12);
    });

    ok &= rc::check("Galerkin on graph Laplacian preserves row-sum-zero", [&] {
        // Path-graph 0-1-2-3, rows sum to zero.
        const std::vector<double> A = {
             1.0, -1.0,  0.0,  0.0,
            -1.0,  2.0, -1.0,  0.0,
             0.0, -1.0,  2.0, -1.0,
             0.0,  0.0, -1.0,  1.0,
        };
        const auto A_csr = dense_to_csr(A, 4);
        const auto Ac    = tls::galerkin_csr(cmap, 2, A_csr);
        const double row0 = mat_at(Ac, 0, 0) + mat_at(Ac, 0, 1);
        const double row1 = mat_at(Ac, 1, 0) + mat_at(Ac, 1, 1);
        RC_ASSERT(std::fabs(row0) < 1e-12);
        RC_ASSERT(std::fabs(row1) < 1e-12);
    });

    return ok ? 0 : 1;
}
