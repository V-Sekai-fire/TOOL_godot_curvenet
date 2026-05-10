// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
//
// C++ mirror of `lean/Curvenet/HierarchicalSparsifyCompensate.lean`.
//
// HSC = Hierarchical Sparsify and Compensate
// (Krishnan-Fattal-Szeliski 2013, `KrishnanFattalSzeliski2013HSC`).
//
// Used as a PCG preconditioner where the multilevel Schwarz / HEM
// family stalled (see docs/IMPOSSIBILITY.md and the TOMBSTONE
// headers in src/curvenet/{two_level,multi_level}_schwarz.h).
// HSC adds the Schur-compensation step those lacked: each
// elimination of a vertex is matched by an adjustment of the
// remaining edge weights so the reduced Laplacian is exactly the
// Schur complement of the original on the kept-vertex subset.
//
// This file ships the load-bearing per-vertex elimination
// primitive plus the level-build/V-cycle pieces are layered on
// top in subsequent cycles.

#ifndef CURVENET_HIERARCHICAL_SPARSIFY_H
#define CURVENET_HIERARCHICAL_SPARSIFY_H

#include <algorithm>
#include <cstddef>
#include <unordered_map>
#include <vector>

#include "sparse_linalg.h"

namespace curvenet {
namespace hsc {

// A symmetric weighted graph, stored as canonicalised edge list
// (u < v) to make Schur compensation straightforward. The Lean
// spec uses the same shape; the C++ runtime path will convert
// to/from `sparse::SparseMatrixCSR` at the boundaries.
struct Graph {
    std::size_t                                          num_verts = 0;
    std::vector<std::tuple<int, int, double>>           edges;     // (u, v, w), u < v
};

// Sum of weights of all edges incident to v.
inline double degree_weight(const Graph &g, int v) {
    double s = 0.0;
    for (const auto &e : g.edges) {
        const int u = std::get<0>(e);
        const int w = std::get<1>(e);
        if (u == v || w == v) s += std::get<2>(e);
    }
    return s;
}

// Edge weight w(a, b), or 0 if no such edge.
inline double edge_weight(const Graph &g, int a, int b) {
    const int lo = std::min(a, b);
    const int hi = std::max(a, b);
    for (const auto &e : g.edges) {
        if (std::get<0>(e) == lo && std::get<1>(e) == hi) return std::get<2>(e);
    }
    return 0.0;
}

// Schur-compensated elimination of vertex v. For every pair of
// v's neighbors (i, j), the new edge weight is:
//   w_new(i, j) = w_old(i, j) + w(v, i) · w(v, j) / deg(v).
// Existing edges accumulate the compensation; missing edges are
// inserted. All edges incident to v are dropped from the output.
inline Graph eliminate_vertex(const Graph &g, int v) {
    // Collect neighbors of v and v's degree weight.
    std::vector<std::pair<int, double>> nbrs;
    double deg_v = 0.0;
    for (const auto &e : g.edges) {
        const int u = std::get<0>(e);
        const int w = std::get<1>(e);
        const double ew = std::get<2>(e);
        if (u == v) { nbrs.push_back({ w, ew }); deg_v += ew; }
        else if (w == v) { nbrs.push_back({ u, ew }); deg_v += ew; }
    }

    // Copy edges that don't touch v.
    Graph out;
    out.num_verts = g.num_verts;
    out.edges.reserve(g.edges.size());
    for (const auto &e : g.edges) {
        const int u = std::get<0>(e);
        const int w = std::get<1>(e);
        if (u != v && w != v) out.edges.push_back(e);
    }

    if (deg_v <= 0.0) return out;

    // Apply compensation across all neighbor pairs.
    for (std::size_t i = 0; i < nbrs.size(); ++i) {
        for (std::size_t j = i + 1; j < nbrs.size(); ++j) {
            const int ni = nbrs[i].first;
            const int nj = nbrs[j].first;
            const double wi = nbrs[i].second;
            const double wj = nbrs[j].second;
            const int lo = std::min(ni, nj);
            const int hi = std::max(ni, nj);
            const double comp = wi * wj / deg_v;
            // Find existing (lo, hi) edge in out, accumulate, or push.
            bool found = false;
            for (auto &e : out.edges) {
                if (std::get<0>(e) == lo && std::get<1>(e) == hi) {
                    std::get<2>(e) += comp;
                    found = true;
                    break;
                }
            }
            if (!found) out.edges.push_back({ lo, hi, comp });
        }
    }
    return out;
}

// Convert a Laplacian SparseMatrixCSR into a HSC Graph by reading
// off-diagonal entries. The Laplacian convention used here is:
//   L[i, i] = sum_j w(i, j)
//   L[i, j] = -w(i, j) for i != j
// so the graph weight is `-L[i, j]`. Self-loops (diagonal) are
// not added to the edge list.
inline Graph csr_to_graph(const sparse::SparseMatrixCSR &A) {
    Graph g;
    g.num_verts = A.rows;
    g.edges.reserve(A.values.size() / 2);
    for (std::size_t i = 0; i < A.rows; ++i) {
        for (int k = A.row_ptr[i]; k < A.row_ptr[i + 1]; ++k) {
            const int j = A.col_idx[k];
            if (static_cast<std::size_t>(j) > i) {
                const double w = -A.values[k];
                g.edges.push_back({ static_cast<int>(i), j, w });
            }
        }
    }
    return g;
}

// Convert a Graph back to a SparseMatrixCSR Laplacian on
// num_verts rows. Each edge contributes `-w` off-diagonally and
// `+w` on both diagonals it touches.
inline sparse::SparseMatrixCSR graph_to_csr(const Graph &g) {
    const std::size_t n = g.num_verts;
    std::vector<std::vector<std::pair<int, double>>> rows(n);
    std::vector<double> diag(n, 0.0);
    for (const auto &e : g.edges) {
        const int u = std::get<0>(e);
        const int v = std::get<1>(e);
        const double w = std::get<2>(e);
        rows[u].push_back({ v, -w });
        rows[v].push_back({ u, -w });
        diag[u] += w;
        diag[v] += w;
    }
    for (std::size_t i = 0; i < n; ++i) {
        rows[i].push_back({ static_cast<int>(i), diag[i] });
        std::sort(rows[i].begin(), rows[i].end(),
                     [](auto &a, auto &b) { return a.first < b.first; });
    }
    sparse::SparseMatrixCSR out;
    out.rows = n;
    out.cols = n;
    out.row_ptr.assign(n + 1, 0);
    for (std::size_t i = 0; i < n; ++i) out.row_ptr[i + 1] = out.row_ptr[i] + static_cast<int>(rows[i].size());
    out.col_idx.resize(out.row_ptr[n]);
    out.values.resize(out.row_ptr[n]);
    for (std::size_t i = 0; i < n; ++i) {
        int p = out.row_ptr[i];
        for (const auto &kv : rows[i]) {
            out.col_idx[p] = kv.first;
            out.values[p] = kv.second;
            ++p;
        }
    }
    return out;
}

} // namespace hsc
} // namespace curvenet

#endif
