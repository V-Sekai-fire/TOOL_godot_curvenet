// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
//
// Tris-to-quads via LEMON's MaxWeightedMatching (Edmonds' blossom algorithm).
//
// The triangle dual graph: one node per input triangle, one edge per shared
// mesh-edge between two triangles. Edge weight = 1 + tiebreak * length / max_length.
// Picking a maximum-weight matching is equivalent to dissolving the largest set
// of mesh edges such that no triangle ends up in two quads.

#include "curvenet/tris_to_quads.h"

#include <lemon/list_graph.h>
#include <lemon/matching.h>

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

namespace curvenet {

namespace {

struct EdgeKey {
	int a;
	int b;

	EdgeKey(int x, int y) : a(std::min(x, y)), b(std::max(x, y)) {}

	bool operator==(const EdgeKey &o) const { return a == o.a && b == o.b; }
};

struct EdgeKeyHash {
	std::size_t operator()(const EdgeKey &e) const noexcept {
		return std::hash<std::int64_t>{}((static_cast<std::int64_t>(e.a) << 32) ^ static_cast<std::int64_t>(e.b));
	}
};

struct CandidateEdge {
	int tri_a;
	int tri_b;
	int shared_v0;
	int shared_v1;
	double length;
};

inline int third_vertex_index(const std::array<int, 3> &t, int sv0, int sv1) {
	for (int i = 0; i < 3; ++i) {
		if (t[i] != sv0 && t[i] != sv1) {
			return i;
		}
	}
	return -1;
}

// CCW quad from two CCW triangles A, B sharing edge {sv0, sv1}.
// If A's "third vertex" sits at index k in A, the quad is
//     (A[k], A[(k+1)%3], B_third, A[(k+2)%3]).
inline PolyFace make_quad(const std::array<int, 3> &A, const std::array<int, 3> &B,
		int sv0, int sv1) {
	const int k_a = third_vertex_index(A, sv0, sv1);
	const int k_b = third_vertex_index(B, sv0, sv1);
	PolyFace q;
	q.count = 4;
	q.v[0] = A[k_a];
	q.v[1] = A[(k_a + 1) % 3];
	q.v[2] = B[k_b];
	q.v[3] = A[(k_a + 2) % 3];
	return q;
}

} // namespace

PolyMesh tris_to_quads(const TriMesh &input, const TrisToQuadsParams &params) {
	PolyMesh out;
	out.vertices = input.vertices;

	const int T = static_cast<int>(input.triangles.size());
	if (T == 0) {
		return out;
	}

	// 1. Edge -> triangles index, with shared endpoints + length.
	std::unordered_map<EdgeKey, std::vector<int>, EdgeKeyHash> edge_to_tris;
	edge_to_tris.reserve(static_cast<std::size_t>(T) * 3);
	for (int i = 0; i < T; ++i) {
		const auto &t = input.triangles[i];
		for (int e = 0; e < 3; ++e) {
			edge_to_tris[EdgeKey(t[e], t[(e + 1) % 3])].push_back(i);
		}
	}

	// 2. Candidate edges: interior + manifold (exactly 2 incident triangles).
	std::vector<CandidateEdge> candidates;
	candidates.reserve(edge_to_tris.size());
	double max_length = 0.0;
	for (const auto &kv : edge_to_tris) {
		if (kv.second.size() != 2) {
			continue;
		}
		const Vec3 &p0 = input.vertices[kv.first.a];
		const Vec3 &p1 = input.vertices[kv.first.b];
		double len = (p1 - p0).length();
		max_length = std::max(max_length, len);
		candidates.push_back({ kv.second[0], kv.second[1], kv.first.a, kv.first.b, len });
	}

	auto emit_unmatched_tris_only = [&]() {
		out.faces.reserve(input.triangles.size());
		for (const auto &t : input.triangles) {
			PolyFace f;
			f.count = 3;
			f.v[0] = t[0];
			f.v[1] = t[1];
			f.v[2] = t[2];
			out.faces.push_back(f);
		}
	};

	if (candidates.empty()) {
		emit_unmatched_tris_only();
		return out;
	}

	// 3. LEMON max-weight matching on the triangle dual graph.
	using Graph = lemon::ListGraph;
	Graph g;
	std::vector<Graph::Node> nodes(T);
	for (int i = 0; i < T; ++i) {
		nodes[i] = g.addNode();
	}
	Graph::EdgeMap<double> weight(g);
	std::vector<Graph::Edge> graph_edges;
	graph_edges.reserve(candidates.size());
	const double inv_max = max_length > 0.0 ? (1.0 / max_length) : 0.0;
	for (const auto &c : candidates) {
		Graph::Edge e = g.addEdge(nodes[c.tri_a], nodes[c.tri_b]);
		weight[e] = 1.0 + params.length_tiebreak * c.length * inv_max;
		graph_edges.push_back(e);
	}

	lemon::MaxWeightedMatching<Graph, Graph::EdgeMap<double>> mwm(g, weight);
	mwm.run();

	// 4. Emit quads from matched candidate edges; unmatched triangles stay tris.
	std::vector<bool> consumed(T, false);
	out.faces.reserve(input.triangles.size());
	for (std::size_t i = 0; i < candidates.size(); ++i) {
		if (!mwm.matching(graph_edges[i])) {
			continue;
		}
		const auto &c = candidates[i];
		consumed[c.tri_a] = true;
		consumed[c.tri_b] = true;
		out.faces.push_back(make_quad(input.triangles[c.tri_a], input.triangles[c.tri_b],
				c.shared_v0, c.shared_v1));
	}
	for (int i = 0; i < T; ++i) {
		if (consumed[i]) {
			continue;
		}
		const auto &t = input.triangles[i];
		PolyFace f;
		f.count = 3;
		f.v[0] = t[0];
		f.v[1] = t[1];
		f.v[2] = t[2];
		out.faces.push_back(f);
	}
	return out;
}

} // namespace curvenet
