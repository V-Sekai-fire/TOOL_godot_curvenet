// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
#ifndef CURVENET_DEFORMER_3D_H
#define CURVENET_DEFORMER_3D_H

#include "curvenet/curvenet.h"
#include "curvenet/tris_to_quads.h"

#include <cstdint>
#include <vector>

#include <godot_cpp/classes/curve3d.hpp>
#include <godot_cpp/classes/mesh_instance3d.hpp>
#include <godot_cpp/variant/node_path.hpp>
#include <godot_cpp/variant/typed_array.hpp>

namespace godot {

// CurveNetDeformer3D drives a source mesh's vertices via Pixar-style profile
// curves. It triangulates the input mesh, fuses pairs of triangles into quads
// via LEMON max-weight matching (so each face has 4 boundaries), then evaluates
// each face as a bilinear-blended Coons patch with the user-authored profile
// curves as boundaries.
//
// Authoring: drop a `CurveNetDeformer3D` next to your mesh and assign a few
// `Curve3D` resources to `profile_curves`. The existing VertexHandles addon
// (sibling repo) gives you in-editor handles for the underlying control points.
class CurveNetDeformer3D : public MeshInstance3D {
	GDCLASS(CurveNetDeformer3D, MeshInstance3D)

	NodePath source_path;
	TypedArray<Curve3D> profile_curves;
	double length_tiebreak = 0.1;
	bool deformation_active = false;

	// Cache of the rest-pose pipeline: tris_to_quads + bind_polymesh outputs.
	// Invalidated when source mesh, source_path, or length_tiebreak changes.
	struct RestCache {
		bool valid = false;
		curvenet::PolyMesh poly;
		std::vector<curvenet::BoundVertex> bindings;
		// Hash of the source ArrayMesh's vertex/index data so we can detect
		// edits without recomputing every frame.
		uint64_t source_hash = 0;
	};
	mutable RestCache rest_cache;

protected:
	static void _bind_methods();
	void invalidate_cache();

public:
	CurveNetDeformer3D() = default;
	~CurveNetDeformer3D() = default;

	void set_source_path(const NodePath &p_path);
	NodePath get_source_path() const;

	void set_profile_curves(const TypedArray<Curve3D> &p_curves);
	TypedArray<Curve3D> get_profile_curves() const;

	void set_length_tiebreak(double p_v);
	double get_length_tiebreak() const;

	void set_deformation_active(bool p_v);
	bool is_deformation_active() const;

	// Recompute the deformed mesh from the current profile_curves and apply
	// it to this MeshInstance3D's mesh. Cheap on subsequent calls because
	// the rest-pose tri->quad fusion + binding is cached.
	void apply_deformation();
};

} // namespace godot

#endif
