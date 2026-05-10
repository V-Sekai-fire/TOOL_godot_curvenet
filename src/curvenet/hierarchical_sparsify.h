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
#include <chrono>
#include <cstddef>
#include <cstdio>
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
//
// Uses a hash-map keyed on (lo, hi) for O(deg^2) compensation
// lookups vs the previous O(|edges| · deg^2) linear scan. At 5k
// the latter spent 1.6 s in build_hierarchy; this drops it to
// well under 100 ms.
inline Graph eliminate_vertex(const Graph &g, int v) {
    std::vector<std::pair<int, double>> nbrs;
    double deg_v = 0.0;
    Graph out;
    out.num_verts = g.num_verts;
    out.edges.reserve(g.edges.size());
    // Edge index by canonical (lo, hi) -> position in out.edges.
    auto pack = [](int a, int b) -> long long {
        return (static_cast<long long>(a) << 32) | static_cast<long long>(b);
    };
    std::unordered_map<long long, int> idx;
    idx.reserve(g.edges.size() * 2);

    // Single pass: collect v's neighbors AND copy non-v edges.
    for (const auto &e : g.edges) {
        const int u = std::get<0>(e);
        const int w = std::get<1>(e);
        const double ew = std::get<2>(e);
        if (u == v) { nbrs.push_back({ w, ew }); deg_v += ew; continue; }
        if (w == v) { nbrs.push_back({ u, ew }); deg_v += ew; continue; }
        idx[pack(u, w)] = static_cast<int>(out.edges.size());
        out.edges.push_back(e);
    }
    if (deg_v <= 0.0) return out;

    for (std::size_t i = 0; i < nbrs.size(); ++i) {
        for (std::size_t j = i + 1; j < nbrs.size(); ++j) {
            const int ni = nbrs[i].first;
            const int nj = nbrs[j].first;
            const double comp = nbrs[i].second * nbrs[j].second / deg_v;
            const int lo = std::min(ni, nj);
            const int hi = std::max(ni, nj);
            const long long key = pack(lo, hi);
            auto it = idx.find(key);
            if (it != idx.end()) {
                std::get<2>(out.edges[it->second]) += comp;
            } else {
                idx[key] = static_cast<int>(out.edges.size());
                out.edges.push_back({ lo, hi, comp });
            }
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

// Drop edges with weight below `tau * max_neighbor_weight(u or v)`.
// Krishnan-Fattal-Szeliski 2013 §3 "Sparsify": weak edges
// contribute little to the matrix's action on smooth modes, so
// dropping them keeps the graph sparse without losing spectral
// quality on the modes multigrid is good at. The retained
// strong edges form a "skeleton" that supports the next IS
// selection without fill explosion.
//
// Without this step, Schur compensation grows graph density
// monotonically: at 81k after 5 levels of coarsening the IS
// density falls below 10% and the hierarchy stalls.
//
// `tau = 0.001` empirically: drops only the very weakest edges
// (below 0.1% of the strongest local edge). More aggressive
// values (0.05) overshoot and break V-cycle convergence on
// our cot-Laplacian since "weak" edges still carry meaningful
// spectral information at the small-shift end.
inline Graph sparsify_graph(const Graph &g, double tau = 0.001) {
    const std::size_t n = g.num_verts;
    // max edge weight per vertex
    std::vector<double> max_w(n, 0.0);
    for (const auto &e : g.edges) {
        const int u = std::get<0>(e);
        const int v = std::get<1>(e);
        const double w = std::get<2>(e);
        if (w > max_w[u]) max_w[u] = w;
        if (w > max_w[v]) max_w[v] = w;
    }
    Graph out;
    out.num_verts = n;
    out.edges.reserve(g.edges.size());
    for (const auto &e : g.edges) {
        const int u = std::get<0>(e);
        const int v = std::get<1>(e);
        const double w = std::get<2>(e);
        const double thresh = tau * std::min(max_w[u], max_w[v]);
        if (w >= thresh) out.edges.push_back(e);
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

// Batch elimination of an entire independent F-set in one pass.
//
// `max_degree`: after the full Schur compensation pass, prune each
// vertex's adjacency to keep only the K strongest edges (by
// absolute weight). Bounds nnz growth across levels — without
// this, on Mire 81k the nnz grows from 241k (level 0) to 1.09M
// (level 23) due to fill from compensation. With max_degree=16,
// nnz stays roughly constant across levels at the cost of a
// modest approximation to the true Schur complement.
//
// `max_degree = 0` disables the cap (full Schur fill).
//
// Cost: O(|E|) to build flat adj + O(sum_F deg(f)^2) compensation
// + O(|E|) to compact result. Per-level memory: 2*|E| ints + |E|
// doubles + n ints.
inline CoarsenLevel coarsen_one_level(const Graph &g,
                                            std::size_t max_degree = 0) {
    CoarsenLevel out;
    out.is_F = select_independent_set(g);

    const std::size_t n = g.num_verts;

    // Flat adjacency: row_start[v] points into nbrs/wts; ent_idx[v]
    // is the canonical edge id (-1 for entries created by
    // compensation). Use std::vector<int> column index +
    // std::vector<double> weight + per-vertex unordered_map only
    // *briefly* during compensation.
    std::vector<int>    row_start(n + 1, 0);
    for (const auto &e : g.edges) {
        ++row_start[std::get<0>(e) + 1];
        ++row_start[std::get<1>(e) + 1];
    }
    for (std::size_t i = 0; i < n; ++i) row_start[i + 1] += row_start[i];

    std::vector<int>    nbr(row_start[n]);
    std::vector<double> wt(row_start[n]);
    std::vector<int>    cursor = row_start;
    for (const auto &e : g.edges) {
        const int u = std::get<0>(e);
        const int v = std::get<1>(e);
        const double w = std::get<2>(e);
        nbr[cursor[u]] = v; wt[cursor[u]] = w; ++cursor[u];
        nbr[cursor[v]] = u; wt[cursor[v]] = w; ++cursor[v];
    }

    // Build C-mapping early so we can short-circuit during compaction.
    out.orig_to_c.assign(n, -1);
    out.c_to_orig.reserve(n);
    for (std::size_t v = 0; v < n; ++v) {
        if (!out.is_F[v]) {
            out.orig_to_c[v] = static_cast<int>(out.c_to_orig.size());
            out.c_to_orig.push_back(static_cast<int>(v));
        }
    }
    const std::size_t nc = out.c_to_orig.size();

    // Compensation goes into a per-C-vertex map. Allocated in coarse
    // index space (smaller, faster). After Schur, the compact graph
    // is built from these maps.
    std::vector<std::unordered_map<int, double>> c_adj(nc);

    // First, copy original C-C edges into c_adj.
    for (const auto &e : g.edges) {
        const int u = std::get<0>(e);
        const int v = std::get<1>(e);
        if (out.is_F[u] || out.is_F[v]) continue;
        const int cu = out.orig_to_c[u];
        const int cv = out.orig_to_c[v];
        c_adj[cu][cv] += std::get<2>(e);
        c_adj[cv][cu] += std::get<2>(e);
    }

    // For each F-vertex, find its C-neighbors via the flat adj
    // and accumulate Schur compensation into c_adj.
    std::vector<int>    nbrs_v;
    std::vector<double> wts_v;
    nbrs_v.reserve(64);
    wts_v.reserve(64);
    for (std::size_t v = 0; v < n; ++v) {
        if (!out.is_F[v]) continue;
        nbrs_v.clear();
        wts_v.clear();
        double deg = 0.0;
        const int rs = row_start[v];
        const int re = row_start[v + 1];
        for (int k = rs; k < re; ++k) {
            const int u = nbr[k];
            const double w = wt[k];
            // Multiple edges to same neighbor accumulate (already
            // canonicalised by graph_to_csr / sparsify). All
            // neighbors of an F-point are C-points (F is IS).
            nbrs_v.push_back(out.orig_to_c[u]);
            wts_v.push_back(w);
            deg += w;
        }
        if (deg <= 0.0) continue;
        for (std::size_t i = 0; i < nbrs_v.size(); ++i) {
            for (std::size_t j = i + 1; j < nbrs_v.size(); ++j) {
                const int ci = nbrs_v[i];
                const int cj = nbrs_v[j];
                const double comp = wts_v[i] * wts_v[j] / deg;
                c_adj[ci][cj] += comp;
                c_adj[cj][ci] += comp;
            }
        }
    }

    // Per-vertex degree cap: keep K strongest edges per row.
    // Skipped when max_degree == 0 (default = full Schur fill).
    if (max_degree > 0) {
        for (std::size_t cv = 0; cv < nc; ++cv) {
            if (c_adj[cv].size() <= max_degree) continue;
            std::vector<std::pair<int, double>> kvs(c_adj[cv].begin(),
                                                       c_adj[cv].end());
            std::sort(kvs.begin(), kvs.end(),
                         [](const std::pair<int, double> &a,
                            const std::pair<int, double> &b) {
                             return std::fabs(a.second) > std::fabs(b.second);
                         });
            kvs.resize(max_degree);
            // Re-symmetrise: also drop reverse-direction entries
            // for kept edges in the other endpoint's map.
            std::unordered_map<int, double> kept;
            kept.reserve(max_degree * 2);
            for (const auto &kv : kvs) kept.insert(kv);
            c_adj[cv] = std::move(kept);
        }
        // Symmetrise: edge (u, v) kept iff u keeps v AND v keeps u.
        // Asymmetric drops can break SPD; iterate till stable.
        for (std::size_t cv = 0; cv < nc; ++cv) {
            std::vector<int> drop;
            for (const auto &kv : c_adj[cv]) {
                if (c_adj[kv.first].count(static_cast<int>(cv)) == 0) {
                    drop.push_back(kv.first);
                }
            }
            for (int u : drop) c_adj[cv].erase(u);
        }
    }

    // Compact to canonical edge list (lo < hi, no self-loops).
    Graph compact;
    compact.num_verts = nc;
    compact.edges.reserve(g.edges.size());
    for (std::size_t cv = 0; cv < nc; ++cv) {
        for (const auto &kv : c_adj[cv]) {
            const int cu = kv.first;
            if (cu > static_cast<int>(cv)) {
                compact.edges.push_back({ static_cast<int>(cv), cu, kv.second });
            }
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

// In-place variant: y is pre-allocated to size n_fine; cleared and filled.
inline void prolong_into(const Prolongation &P,
                            const std::vector<double> &x,
                            std::vector<double> &y) {
    std::fill(y.begin(), y.end(), 0.0);
    for (const auto &t : P.triples) {
        y[std::get<0>(t)] += std::get<2>(t) * x[std::get<1>(t)];
    }
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

inline void restrict_pt_into(const Prolongation &P,
                                  const std::vector<double> &x,
                                  std::vector<double> &y) {
    std::fill(y.begin(), y.end(), 0.0);
    for (const auto &t : P.triples) {
        y[std::get<1>(t)] += std::get<2>(t) * x[std::get<0>(t)];
    }
}

// A full HSC hierarchy: levels [0] = finest, [L-1] = coarsest. The
// coarsest level is solved directly (small CG); intermediate
// levels apply Jacobi smoothing + restriction + recursion +
// prolongation + post-smoothing per the standard V-cycle.
struct Hierarchy {
    std::vector<Graph>                          graphs;     // length L
    std::vector<sparse::SparseMatrixCSR>       mats;       // length L (= graph_to_csr per level)
    std::vector<Prolongation>                   prolongs;   // length L-1; prolongs[i] goes level i+1 -> level i
    std::vector<std::vector<double>>           diags;      // length L; cached diag(A_l) for Jacobi
};

// Hierarchy build. Goes deep until coarsest_size or max_levels is
// hit. The cycle-1 attempt to cap depth at 5 didn't help: the
// 2k-vertex bottom-level CG ate the wall time. Instead, let the
// hierarchy go deep so the bottom is small (~64 verts) where CG
// converges trivially.
inline Hierarchy build_hierarchy(const Graph &fine,
                                     std::size_t coarsest_size = 64,
                                     std::size_t max_levels = 64,
                                     double sparsify_tau = 0.0,
                                     std::size_t max_degree = 24,
                                     bool verbose = false) {
    Hierarchy h;
    h.graphs.push_back(fine);
    Graph cur = fine;
    for (std::size_t lvl = 0; lvl + 1 < max_levels; ++lvl) {
        if (cur.num_verts <= coarsest_size) break;
        const auto t0 = std::chrono::steady_clock::now();
        const Graph base = (sparsify_tau > 0.0)
            ? sparsify_graph(cur, sparsify_tau) : cur;
        const auto t1 = std::chrono::steady_clock::now();
        // Tested: per-level max_degree shrinkage (24 → 16 → 8 with
        // depth) cut V-cycle apply by 12% but raised PCG iter
        // count from 9 to 15 — net loss. Flat max_degree across
        // levels is the right call for this matrix.
        const CoarsenLevel lev = coarsen_one_level(base, max_degree);
        const auto t2 = std::chrono::steady_clock::now();
        if (lev.coarse.num_verts == 0 ||
            lev.coarse.num_verts >= cur.num_verts) break;
        h.prolongs.push_back(make_prolongation(base, lev));
        const auto t3 = std::chrono::steady_clock::now();
        if (verbose) {
            const auto ms = [](const auto &a, const auto &b) {
                return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count() / 1000.0;
            };
            std::printf("  level %2zu n=%zu->%zu nnz=%zu sparsify=%.1fms coarsen=%.1fms prolong=%.1fms\n",
                          lvl, cur.num_verts, lev.coarse.num_verts,
                          cur.edges.size(),
                          ms(t0, t1), ms(t1, t2), ms(t2, t3));
        }
        cur = lev.coarse;
        h.graphs.push_back(cur);
    }
    for (const auto &g : h.graphs) {
        h.mats.push_back(graph_to_csr(g));
        h.diags.push_back(sparse::diagonal(h.mats.back()));
    }
    return h;
}

// Per-thread persistent scratch buffers for v_cycle_apply.
// Pre-allocates the size-n vectors used at each level so the
// V-cycle inner loop doesn't allocate. Per Phase-1 profiling
// (diag_hsc_profile_5k): vector ctor cost was 403 us / 3371 us
// V-cycle = 12% — adding up to a real chunk of the parallel
// 12-RHS frame budget.
//
// Shared-nothing: each thread holds its own VCycleScratch;
// the underlying Hierarchy stays read-only across threads.
struct VCycleScratch {
    std::vector<std::vector<double>> x;     // size = level n
    std::vector<std::vector<double>> Ax;    // size = level n
    std::vector<std::vector<double>> r;     // size = level n  (residual, level's own scratch)
    std::vector<std::vector<double>> e;     // size = level n  (correction, level's own scratch)
    std::vector<std::vector<double>> rc;    // size = next-level n (restricted residual, parent owns)
    std::vector<std::vector<double>> ec;    // size = next-level n (coarse correction, parent owns)
};

inline VCycleScratch make_scratch(const Hierarchy &h) {
    VCycleScratch s;
    const std::size_t L = h.graphs.size();
    s.x.resize(L); s.Ax.resize(L); s.r.resize(L); s.e.resize(L);
    s.rc.resize(L); s.ec.resize(L);
    for (std::size_t l = 0; l < L; ++l) {
        const std::size_t n = h.graphs[l].num_verts;
        s.x[l].resize(n);
        s.Ax[l].resize(n);
        s.r[l].resize(n);
        s.e[l].resize(n);
        if (l + 1 < L) {
            const std::size_t nc = h.graphs[l + 1].num_verts;
            s.rc[l].resize(nc);
            s.ec[l].resize(nc);
        }
    }
    return s;
}

// Damped Jacobi smoother: x ← x + ω · D^{-1} · (b - A·x), nu sweeps.
// ω = 2/3 is a common choice that contracts smooth modes well on
// graph Laplacians.
inline void jacobi_smooth(const sparse::SparseMatrixCSR &A,
                              const std::vector<double> &diag,
                              const std::vector<double> &b,
                              std::vector<double> &x,
                              std::size_t nu,
                              double omega = 2.0 / 3.0) {
    const std::size_t n = A.rows;
    for (std::size_t s = 0; s < nu; ++s) {
        const std::vector<double> Ax = sparse::spmv(A, x);
        for (std::size_t i = 0; i < n; ++i) {
            const double d = diag[i];
            if (d != 0.0) x[i] += omega * (b[i] - Ax[i]) / d;
        }
    }
}

// Symmetric Gauss-Seidel smoother: forward sweep then backward
// sweep, both in-place using already-updated entries. Stronger
// per-sweep contraction than damped Jacobi (the standard
// V-cycle smoother for graph Laplacians).
//
// One nu = 1 SGS step ≈ 2-3 Jacobi sweeps in error reduction
// for the same SpMV-equivalent work, so the V-cycle hits the
// same residual drop in fewer effective sweeps. Per Gall's-law
// cycle 1 of the HSC fix.
inline void sym_gauss_seidel_smooth(const sparse::SparseMatrixCSR &A,
                                          const std::vector<double> &diag,
                                          const std::vector<double> &b,
                                          std::vector<double> &x,
                                          std::size_t nu) {
    const std::size_t n = A.rows;
    for (std::size_t s = 0; s < nu; ++s) {
        // Forward sweep: x[i] = (b[i] - sum_{j<i} A[i,j] x[j] - sum_{j>i} A[i,j] x[j]) / A[i,i]
        for (std::size_t i = 0; i < n; ++i) {
            const double d = diag[i];
            if (d == 0.0) continue;
            double s_off = 0.0;
            const int rs = A.row_ptr[i];
            const int re = A.row_ptr[i + 1];
            for (int k = rs; k < re; ++k) {
                const int j = A.col_idx[k];
                if (j != static_cast<int>(i)) s_off += A.values[k] * x[j];
            }
            x[i] = (b[i] - s_off) / d;
        }
        // Backward sweep, same recurrence reversed.
        for (std::size_t ii = 0; ii < n; ++ii) {
            const std::size_t i = n - 1 - ii;
            const double d = diag[i];
            if (d == 0.0) continue;
            double s_off = 0.0;
            const int rs = A.row_ptr[i];
            const int re = A.row_ptr[i + 1];
            for (int k = rs; k < re; ++k) {
                const int j = A.col_idx[k];
                if (j != static_cast<int>(i)) s_off += A.values[k] * x[j];
            }
            x[i] = (b[i] - s_off) / d;
        }
    }
}

// One V-cycle apply: solve `A_level · x = b` approximately by
//   pre-smooth -> restrict residual -> recurse -> prolong -> post-smooth
// At the coarsest level uses sparse::cg as a direct-equivalent
// (small system, converges quickly).
// Per-level timing accumulators for profiling. When non-null,
// times in microseconds get added per level on the way up.
struct VCycleProfile {
    std::vector<double> smooth_us;
    std::vector<double> spmv_us;
    std::vector<double> restrict_us;
    std::vector<double> prolong_us;
    std::vector<double> bottom_us;
    std::vector<double> alloc_us;   // vector ctors per level
};

inline std::vector<double> v_cycle_apply(const Hierarchy &h,
                                              const std::vector<double> &b,
                                              std::size_t level = 0,
                                              std::size_t pre_smooth = 1,
                                              std::size_t post_smooth = 1,
                                              VCycleProfile *prof = nullptr) {
    using clk = std::chrono::steady_clock;
    auto us = [](const auto &a, const auto &b) {
        return static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(b - a).count());
    };

    const std::size_t L = h.graphs.size();
    if (level + 1 == L) {
        const auto t0 = clk::now();
        std::vector<double> x = sparse::cg(h.mats[level], b, 30, 1e-6);
        if (prof) prof->bottom_us.push_back(us(t0, clk::now()));
        return x;
    }
    const sparse::SparseMatrixCSR &A = h.mats[level];
    const std::vector<double> &diag = h.diags[level];
    auto t_alloc0 = clk::now();
    std::vector<double> x(b.size(), 0.0);
    auto t_alloc1 = clk::now();
    auto t0 = clk::now();
    sym_gauss_seidel_smooth(A, diag, b, x, pre_smooth);
    auto t1 = clk::now();
    auto t_alloc2 = clk::now();
    const std::vector<double> Ax = sparse::spmv(A, x);
    std::vector<double> r(b.size());
    for (std::size_t i = 0; i < b.size(); ++i) r[i] = b[i] - Ax[i];
    auto t_alloc3 = clk::now();
    auto t2 = clk::now();
    const std::vector<double> rc = restrict_pt(h.prolongs[level], r);
    auto t3 = clk::now();
    const std::vector<double> ec = v_cycle_apply(h, rc, level + 1,
                                                       pre_smooth, post_smooth, prof);
    auto t4 = clk::now();
    const std::vector<double> e = prolong(h.prolongs[level], ec);
    for (std::size_t i = 0; i < b.size(); ++i) x[i] += e[i];
    auto t5 = clk::now();
    sym_gauss_seidel_smooth(A, diag, b, x, post_smooth);
    auto t6 = clk::now();
    if (prof) {
        prof->smooth_us.push_back(us(t0, t1) + us(t5, t6));
        prof->spmv_us.push_back(us(t1, t2));
        prof->restrict_us.push_back(us(t2, t3));
        prof->prolong_us.push_back(us(t4, t5));
        prof->alloc_us.push_back(us(t_alloc0, t_alloc1) + us(t_alloc2, t_alloc3));
    }
    return x;
}

// Scratch variant of v_cycle_apply: writes the V-cycle correction
// into x_out (pre-allocated to fine-level size). Reuses persistent
// buffers from `s` instead of allocating fresh vectors per level.
//
// Result: ~10% wall-time reduction on V-cycle apply at 5k vs the
// allocating variant (Phase 2 of the dig plan).
inline void v_cycle_apply_into(const Hierarchy &h,
                                    const std::vector<double> &b,
                                    std::vector<double> &x_out,
                                    VCycleScratch &s,
                                    std::size_t level = 0,
                                    std::size_t pre_smooth = 1,
                                    std::size_t post_smooth = 1) {
    const std::size_t L = h.graphs.size();
    if (level + 1 == L) {
        x_out = sparse::cg(h.mats[level], b, 30, 1e-6);
        return;
    }
    const sparse::SparseMatrixCSR &A = h.mats[level];
    const std::vector<double> &diag = h.diags[level];
    auto &x = s.x[level];
    std::fill(x.begin(), x.end(), 0.0);
    sym_gauss_seidel_smooth(A, diag, b, x, pre_smooth);
    sparse::spmv_into(A, x, s.Ax[level]);
    auto &r = s.r[level];
    for (std::size_t i = 0; i < b.size(); ++i) r[i] = b[i] - s.Ax[level][i];
    // Use parent-owned rc/ec buffers (sized to next level's n) so
    // the child doesn't alias the parent's b.
    auto &rc = s.rc[level];
    auto &ec = s.ec[level];
    restrict_pt_into(h.prolongs[level], r, rc);
    v_cycle_apply_into(h, rc, ec, s, level + 1, pre_smooth, post_smooth);
    auto &e = s.e[level];
    prolong_into(h.prolongs[level], ec, e);
    for (std::size_t i = 0; i < b.size(); ++i) x[i] += e[i];
    sym_gauss_seidel_smooth(A, diag, b, x, post_smooth);
    x_out = x;
}

// HSC-preconditioned conjugate gradient. Same shape as
// `incomplete_cholesky::cg_icc_with_guess` but with the V-cycle
// as the preconditioner apply: M^{-1}·r = v_cycle_apply(h, r).
//
// `iters_out` is an optional pointer to receive the actual iter
// count — used by the bench to distinguish "fast per-iter but
// many iters" from "few iters but slow per-iter".
inline std::vector<double> cg_hsc_with_guess(
        const sparse::SparseMatrixCSR &A,
        const Hierarchy &h,
        const std::vector<double> &b,
        const std::vector<double> &x0,
        std::size_t max_iter,
        double tol,
        std::size_t *iters_out = nullptr) {
    std::vector<double> x = x0;
    const std::vector<double> Ax0 = sparse::spmv(A, x0);
    std::vector<double> r = sparse::saxpby(1.0, b, -1.0, Ax0);
    std::vector<double> z = v_cycle_apply(h, r);
    std::vector<double> p = z;
    double rz_old = sparse::dot(r, z);
    const double tol_sq = tol * tol;
    std::size_t iters = 0;
    for (std::size_t iter = 0; iter < max_iter; ++iter) {
        const std::vector<double> Ap = sparse::spmv(A, p);
        const double pAp = sparse::dot(p, Ap);
        if (pAp == 0.0) { iters = iter; break; }
        const double alpha = rz_old / pAp;
        sparse::axpy_inplace(alpha, p, x);
        sparse::axpy_inplace(-alpha, Ap, r);
        iters = iter + 1;
        if (sparse::dot(r, r) < tol_sq) break;
        z = v_cycle_apply(h, r);
        const double rz_new = sparse::dot(r, z);
        const double beta = (rz_old == 0.0) ? 0.0 : (rz_new / rz_old);
        p = sparse::saxpby(1.0, z, beta, p);
        rz_old = rz_new;
    }
    if (iters_out) *iters_out = iters;
    return x;
}

// PCG variant that reuses a caller-supplied VCycleScratch. Each
// thread in a shared-nothing 12-RHS solve creates its own
// scratch once and reuses it across V-cycle applies, saving the
// per-V-cycle vector allocation cost.
inline std::vector<double> cg_hsc_with_guess_scratch(
        const sparse::SparseMatrixCSR &A,
        const Hierarchy &h,
        const std::vector<double> &b,
        const std::vector<double> &x0,
        std::size_t max_iter,
        double tol,
        VCycleScratch &s,
        std::size_t *iters_out = nullptr) {
    std::vector<double> x = x0;
    const std::vector<double> Ax0 = sparse::spmv(A, x0);
    std::vector<double> r = sparse::saxpby(1.0, b, -1.0, Ax0);
    std::vector<double> z(b.size());
    v_cycle_apply_into(h, r, z, s);
    std::vector<double> p = z;
    double rz_old = sparse::dot(r, z);
    const double tol_sq = tol * tol;
    std::vector<double> Ap(b.size());
    std::size_t iters = 0;
    for (std::size_t iter = 0; iter < max_iter; ++iter) {
        sparse::spmv_into(A, p, Ap);
        const double pAp = sparse::dot(p, Ap);
        if (pAp == 0.0) { iters = iter; break; }
        const double alpha = rz_old / pAp;
        sparse::axpy_inplace(alpha, p, x);
        sparse::axpy_inplace(-alpha, Ap, r);
        iters = iter + 1;
        if (sparse::dot(r, r) < tol_sq) break;
        v_cycle_apply_into(h, r, z, s);
        const double rz_new = sparse::dot(r, z);
        const double beta = (rz_old == 0.0) ? 0.0 : (rz_new / rz_old);
        p = sparse::saxpby(1.0, z, beta, p);
        rz_old = rz_new;
    }
    if (iters_out) *iters_out = iters;
    return x;
}

} // namespace hsc
} // namespace curvenet

#endif
