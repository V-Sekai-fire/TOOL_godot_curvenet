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

// Greedy maximum-independent-set selection of "F-points" — the
// vertices to eliminate at this level. A vertex is selected as
// F when it has no F-neighbors yet; all its neighbors are then
// marked as C-points (kept at the next level).
//
// Walks vertices in degree-ascending order so that the densest
// vertices stay as C-points and sparse vertices get eliminated —
// gives well-conditioned coarse problems on triangle meshes.
//
// Returns a vector<char> (acts as bool) of length num_verts:
//   1 = F-point (will be eliminated)
//   0 = C-point (will be kept at next level)
inline std::vector<char> select_independent_set(const Graph &g) {
    const std::size_t n = g.num_verts;
    // Build adjacency lists.
    std::vector<std::vector<int>> adj(n);
    for (const auto &e : g.edges) {
        adj[std::get<0>(e)].push_back(std::get<1>(e));
        adj[std::get<1>(e)].push_back(std::get<0>(e));
    }
    // Sort vertex order by degree ascending for greedy IS.
    std::vector<int> order(n);
    for (std::size_t i = 0; i < n; ++i) order[i] = static_cast<int>(i);
    std::sort(order.begin(), order.end(),
                 [&](int a, int b) { return adj[a].size() < adj[b].size(); });

    std::vector<char> mark(n, 0);   // 0 = unmarked, 1 = F, 2 = C
    for (int v : order) {
        if (mark[v] != 0) continue;
        // No F-neighbor by definition (F's neighbors are marked 2).
        mark[v] = 1;   // F
        for (int nb : adj[v]) if (mark[nb] == 0) mark[nb] = 2;   // C
    }
    std::vector<char> is_F(n, 0);
    for (std::size_t i = 0; i < n; ++i) is_F[i] = (mark[i] == 1) ? 1 : 0;
    return is_F;
}

// One coarsening level: starting from `g`, eliminate every F-point
// (in the order they appear in `is_F`) via Schur compensation,
// returning a coarsened graph that still uses the original
// vertex indexing (eliminated vertices remain in num_verts but
// have no incident edges) plus a list of the surviving C-points
// in original-index order.
//
// Because F is an independent set, eliminating F-points in any
// order yields the same Schur-complement Laplacian on C-points.
struct CoarsenLevel {
    Graph                coarse;       // Laplacian on C-points (still indexed in original space)
    std::vector<int>     c_to_orig;    // length = #C-points; coarse-index -> original-index
    std::vector<int>     orig_to_c;    // length = num_verts;  original-index -> coarse-index, or -1
    std::vector<char>    is_F;         // saved for prolongation construction
};

inline CoarsenLevel coarsen_one_level(const Graph &g) {
    CoarsenLevel out;
    out.is_F = select_independent_set(g);
    Graph cur = g;
    for (std::size_t v = 0; v < g.num_verts; ++v) {
        if (out.is_F[v]) cur = eliminate_vertex(cur, static_cast<int>(v));
    }
    // Build the C-mapping.
    out.orig_to_c.assign(g.num_verts, -1);
    out.c_to_orig.reserve(g.num_verts);
    for (std::size_t v = 0; v < g.num_verts; ++v) {
        if (!out.is_F[v]) {
            out.orig_to_c[v] = static_cast<int>(out.c_to_orig.size());
            out.c_to_orig.push_back(static_cast<int>(v));
        }
    }
    // Compact `cur`: rename vertex indices to the C-only space and
    // strip out edges that touch (now-isolated) F-points.
    Graph compact;
    compact.num_verts = out.c_to_orig.size();
    compact.edges.reserve(cur.edges.size());
    for (const auto &e : cur.edges) {
        const int a = std::get<0>(e);
        const int b = std::get<1>(e);
        const int ca = out.orig_to_c[a];
        const int cb = out.orig_to_c[b];
        if (ca >= 0 && cb >= 0 && ca != cb) {
            compact.edges.push_back({ std::min(ca, cb), std::max(ca, cb), std::get<2>(e) });
        }
    }
    out.coarse = std::move(compact);
    return out;
}

// Sparse prolongation matrix from C-points (coarse, n_c) to all
// vertices (fine, n_f). Stored as a list of (fine_idx, coarse_idx,
// weight) triples. For a C-point v: one entry (v, c_of(v), 1.0).
// For an F-point v: one entry per C-neighbor c, with weight
// w(v, c) / sum of w(v, c') over C-neighbors.
struct Prolongation {
    std::size_t                                  n_fine = 0;
    std::size_t                                  n_coarse = 0;
    std::vector<std::tuple<int, int, double>>   triples;  // (fine, coarse, w)
};

inline Prolongation make_prolongation(const Graph &g, const CoarsenLevel &lev) {
    Prolongation P;
    P.n_fine = g.num_verts;
    P.n_coarse = lev.c_to_orig.size();
    // Build adjacency.
    std::vector<std::vector<std::pair<int, double>>> adj(g.num_verts);
    for (const auto &e : g.edges) {
        const int u = std::get<0>(e);
        const int v = std::get<1>(e);
        const double w = std::get<2>(e);
        adj[u].push_back({ v, w });
        adj[v].push_back({ u, w });
    }
    for (std::size_t v = 0; v < g.num_verts; ++v) {
        if (!lev.is_F[v]) {
            P.triples.push_back({ static_cast<int>(v), lev.orig_to_c[v], 1.0 });
        } else {
            double sum_c = 0.0;
            for (auto &kv : adj[v]) if (!lev.is_F[kv.first]) sum_c += kv.second;
            if (sum_c == 0.0) continue;   // orphan F-point: leave at 0
            for (auto &kv : adj[v]) {
                if (!lev.is_F[kv.first]) {
                    P.triples.push_back({
                        static_cast<int>(v),
                        lev.orig_to_c[kv.first],
                        kv.second / sum_c
                    });
                }
            }
        }
    }
    return P;
}

// y = P · x (size n_fine), x is size n_coarse.
inline std::vector<double> prolong(const Prolongation &P,
                                       const std::vector<double> &x) {
    std::vector<double> y(P.n_fine, 0.0);
    for (const auto &t : P.triples) {
        y[std::get<0>(t)] += std::get<2>(t) * x[std::get<1>(t)];
    }
    return y;
}

// y = P^T · x (size n_coarse), x is size n_fine.
inline std::vector<double> restrict_pt(const Prolongation &P,
                                            const std::vector<double> &x) {
    std::vector<double> y(P.n_coarse, 0.0);
    for (const auto &t : P.triples) {
        y[std::get<1>(t)] += std::get<2>(t) * x[std::get<0>(t)];
    }
    return y;
}

} // namespace hsc
} // namespace curvenet

#endif
