// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
#ifndef CURVENET_HALFEDGE_H
#define CURVENET_HALFEDGE_H

// C++ mirror of `lean/Curvenet/Halfedge.lean`. Defines the halfedge mesh
// data structure used by the DeGoes22 cut-mesh algorithm + the
// `is_manifold` checker that the runtime uses to assert the §4.1
// pre-condition. fTetWild's output (per todos/06) is what feeds this
// invariant in production; authored Godot meshes (PlaneMesh etc.) pass
// it for free.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace curvenet {

// Optional integer (matches Lean's `Option Nat` for halfedge indices and
// face ids). -1 sentinel encodes `none`.
struct OptionalIndex {
	int value = -1;

	OptionalIndex() = default;
	explicit OptionalIndex(int v) : value(v) {}

	bool has_value() const { return value >= 0; }
	int  unwrap()    const { return value; }

	static OptionalIndex none()              { return OptionalIndex(); }
	static OptionalIndex some(int v)         { return OptionalIndex(v); }
	bool   operator==(const OptionalIndex &o) const { return value == o.value; }
	bool   operator!=(const OptionalIndex &o) const { return value != o.value; }
};

struct Halfedge {
	int           target = 0;        // target vertex (head)
	OptionalIndex twin   = OptionalIndex::none();
	int           next   = 0;        // next halfedge in face loop, CCW
	OptionalIndex face   = OptionalIndex::none();
};

struct HalfedgeMesh {
	std::size_t           vertex_count = 0;
	std::size_t           face_count   = 0;
	std::vector<Halfedge> halfedges;

	std::size_t he_count() const { return halfedges.size(); }
};

namespace halfedge_invariants {

// All target / next / twin / face indices fit within their declared ranges.
inline bool indices_in_range(const HalfedgeMesh &m) {
	const std::size_t n = m.he_count();
	for (std::size_t i = 0; i < n; ++i) {
		const Halfedge &h = m.halfedges[i];
		if (h.target < 0 || static_cast<std::size_t>(h.target) >= m.vertex_count) {
			return false;
		}
		if (h.next < 0 || static_cast<std::size_t>(h.next) >= n) {
			return false;
		}
		if (h.twin.has_value()) {
			const int t = h.twin.unwrap();
			if (t < 0 || static_cast<std::size_t>(t) >= n) {
				return false;
			}
		}
		if (h.face.has_value()) {
			const int f = h.face.unwrap();
			if (f < 0 || static_cast<std::size_t>(f) >= m.face_count) {
				return false;
			}
		}
	}
	return true;
}

// twin(twin h) = some h, where defined.
inline bool twin_involutive(const HalfedgeMesh &m) {
	const std::size_t n = m.he_count();
	for (std::size_t i = 0; i < n; ++i) {
		const OptionalIndex t = m.halfedges[i].twin;
		if (!t.has_value()) {
			continue;
		}
		const OptionalIndex tt = m.halfedges[t.unwrap()].twin;
		if (!tt.has_value() || static_cast<std::size_t>(tt.unwrap()) != i) {
			return false;
		}
	}
	return true;
}

// source(h) = target(twin h). Computed by walking face loops to find each
// halfedge's predecessor (the halfedge whose `next` equals it), whose
// target is then the source of h.
inline bool twins_are_opposite(const HalfedgeMesh &m) {
	const std::size_t n = m.he_count();
	std::vector<int> prev(n, 0);
	for (std::size_t i = 0; i < n; ++i) {
		prev[m.halfedges[i].next] = static_cast<int>(i);
	}
	for (std::size_t i = 0; i < n; ++i) {
		const OptionalIndex t = m.halfedges[i].twin;
		if (!t.has_value()) {
			continue;
		}
		const int src_h = m.halfedges[prev[i]].target;
		const int tgt_t = m.halfedges[t.unwrap()].target;
		if (src_h != tgt_t) {
			return false;
		}
	}
	return true;
}

// Following `.next` from each halfedge of an interior face returns to the
// starting halfedge in at most n steps, all visited halfedges sharing the
// same face index. Boundary halfedges (face = none) are skipped — they
// form a separate loop checked implicitly by twin_involutive.
inline bool face_loops_close(const HalfedgeMesh &m) {
	const std::size_t n = m.he_count();
	for (std::size_t start = 0; start < n; ++start) {
		const OptionalIndex f0 = m.halfedges[start].face;
		if (!f0.has_value()) {
			continue;
		}
		std::size_t cur = m.halfedges[start].next;
		std::size_t steps = 0;
		bool closed = false;
		while (steps < n) {
			if (cur == start) {
				closed = true;
				break;
			}
			if (m.halfedges[cur].face != f0) {
				return false;
			}
			cur = m.halfedges[cur].next;
			++steps;
		}
		if (!closed) {
			return false;
		}
	}
	return true;
}

} // namespace halfedge_invariants

inline bool is_manifold(const HalfedgeMesh &m) {
	using namespace halfedge_invariants;
	return indices_in_range(m) &&
	       twin_involutive(m) &&
	       twins_are_opposite(m) &&
	       face_loops_close(m);
}

} // namespace curvenet

#endif
