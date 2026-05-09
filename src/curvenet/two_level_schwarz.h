// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
//
// TOMBSTONE [loop 100, 2026-05-09]
// 5k converges in 193 iters (slower than 1-level Chebyshev's 137).
// 81k stalls at L_inf residual ~3.7 — same plateau every multilevel
// variant hits. Root cause is the matrix's 7-decade diagonal spread,
// not the level count. See tests/diag_70k_cg_baseline.cpp and
// PERF_BASELINE.md "Dead ends". The Galerkin / restrict / prolong
// helpers below are still reused by multi_level_schwarz.h.
//
// ----
//
// C++ mirror of `lean/Curvenet/TwoLevelSchwarz.lean`.
//
// 1-level Schwarz with 1-ring overlap stalls at the deformer's PCVR-
// target mesh (81k verts, 368 meshlets) because info has to walk
// hundreds of meshlets at one ring per outer iter — measured rate
// ~0.99973/iter. Standard fix per Smith-Bjørstad-Gropp 1996 / Wu
// 2022: add a coarse correction. Each meshlet contributes one
// coarse "node"; the coarse problem A_c · y_c = R · r is small
// enough to solve directly (368 × 368), and prolonging y_c back to
// the fine grid corrects the global low-frequency modes the
// 1-level Schwarz can't reach.

#ifndef CURVENET_TWO_LEVEL_SCHWARZ_H
#define CURVENET_TWO_LEVEL_SCHWARZ_H

#include <cstddef>
#include <vector>

#include "sparse_linalg.h"

namespace curvenet {
namespace two_level_schwarz {

// Each entry `coarse_of[i]` is the coarse-node id (typically the
// owning meshlet) for fine vertex `i`.

// Restrict fine vector to coarse: c[meshlet(i)] += f[i].
inline std::vector<double> restrict_fine(const std::vector<int> &coarse_of,
                                            std::size_t num_coarse,
                                            const std::vector<double> &f) {
    std::vector<double> c(num_coarse, 0.0);
    for (std::size_t i = 0; i < f.size(); ++i) {
        c[static_cast<std::size_t>(coarse_of[i])] += f[i];
    }
    return c;
}

// Prolong coarse vector to fine: f[i] = c[meshlet(i)].
inline std::vector<double> prolong_coarse(const std::vector<int> &coarse_of,
                                            std::size_t num_fine,
                                            const std::vector<double> &c) {
    std::vector<double> f(num_fine, 0.0);
    for (std::size_t i = 0; i < num_fine; ++i) {
        f[i] = c[static_cast<std::size_t>(coarse_of[i])];
    }
    return f;
}

// Cluster sizes (member counts per coarse node).
inline std::vector<std::size_t> coarse_sizes(const std::vector<int> &coarse_of,
                                                std::size_t num_coarse) {
    std::vector<std::size_t> s(num_coarse, 0);
    for (std::size_t i = 0; i < coarse_of.size(); ++i) {
        ++s[static_cast<std::size_t>(coarse_of[i])];
    }
    return s;
}

// Galerkin coarse-matrix: A_c = R · A · Rᵀ, equivalently
//   A_c[c1, c2] = Σ_{i, j with coarse_of[i]=c1, coarse_of[j]=c2} A[i, j]
//
// Built directly from the sparse fine matrix to avoid a dense
// O(n²) intermediate. Output is CSR.
inline sparse::SparseMatrixCSR
galerkin_csr(const std::vector<int> &coarse_of,
              std::size_t num_coarse,
              const sparse::SparseMatrixCSR &A) {
    // Accumulate (c1, c2) → Σ A[i, j] in a dense num_coarse×num_coarse
    // buffer (small — typically 368x368 for our 81k case = 1 MB).
    std::vector<double> dense(num_coarse * num_coarse, 0.0);
    for (std::size_t i = 0; i < A.rows; ++i) {
        const std::size_t c1 = static_cast<std::size_t>(coarse_of[i]);
        const int rs = A.row_ptr[i];
        const int re = A.row_ptr[i + 1];
        for (int k = rs; k < re; ++k) {
            const std::size_t j = static_cast<std::size_t>(A.col_idx[k]);
            const std::size_t c2 = static_cast<std::size_t>(coarse_of[j]);
            dense[c1 * num_coarse + c2] += A.values[k];
        }
    }

    // Convert to CSR.
    sparse::SparseMatrixCSR out;
    out.rows = num_coarse;
    out.cols = num_coarse;
    out.row_ptr.assign(num_coarse + 1, 0);
    for (std::size_t i = 0; i < num_coarse; ++i) {
        for (std::size_t j = 0; j < num_coarse; ++j) {
            if (dense[i * num_coarse + j] != 0.0) {
                ++out.row_ptr[i + 1];
            }
        }
    }
    for (std::size_t i = 0; i < num_coarse; ++i) {
        out.row_ptr[i + 1] += out.row_ptr[i];
    }
    out.col_idx.resize(out.row_ptr.back());
    out.values.resize(out.row_ptr.back());
    std::vector<int> cursor = out.row_ptr;
    for (std::size_t i = 0; i < num_coarse; ++i) {
        for (std::size_t j = 0; j < num_coarse; ++j) {
            const double v = dense[i * num_coarse + j];
            if (v != 0.0) {
                const int idx = cursor[i]++;
                out.col_idx[idx] = static_cast<int>(j);
                out.values[idx]  = v;
            }
        }
    }
    return out;
}

} // namespace two_level_schwarz
} // namespace curvenet

#endif
