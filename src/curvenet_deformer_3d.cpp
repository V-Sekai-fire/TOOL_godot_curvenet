// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
#include "curvenet_deformer_3d.h"

#include "curvenet/tris_to_quads.h"

#include <godot_cpp/classes/array_mesh.hpp>
#include <godot_cpp/classes/mesh.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

namespace godot {

void CurveNetDeformer3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_source_path", "path"), &CurveNetDeformer3D::set_source_path);
	ClassDB::bind_method(D_METHOD("get_source_path"), &CurveNetDeformer3D::get_source_path);
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "source_path"), "set_source_path", "get_source_path");

	ClassDB::bind_method(D_METHOD("set_profile_curves", "curves"), &CurveNetDeformer3D::set_profile_curves);
	ClassDB::bind_method(D_METHOD("get_profile_curves"), &CurveNetDeformer3D::get_profile_curves);
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "profile_curves", PROPERTY_HINT_ARRAY_TYPE,
							  String::num_int64(Variant::OBJECT) + "/" + String::num_int64(PROPERTY_HINT_RESOURCE_TYPE) + ":Curve3D"),
			"set_profile_curves", "get_profile_curves");

	ClassDB::bind_method(D_METHOD("set_length_tiebreak", "v"), &CurveNetDeformer3D::set_length_tiebreak);
	ClassDB::bind_method(D_METHOD("get_length_tiebreak"), &CurveNetDeformer3D::get_length_tiebreak);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "length_tiebreak", PROPERTY_HINT_RANGE, "0.0,1.0,0.01"),
			"set_length_tiebreak", "get_length_tiebreak");

	ClassDB::bind_method(D_METHOD("set_deformation_active", "v"), &CurveNetDeformer3D::set_deformation_active);
	ClassDB::bind_method(D_METHOD("is_deformation_active"), &CurveNetDeformer3D::is_deformation_active);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "deformation_active"),
			"set_deformation_active", "is_deformation_active");

	ClassDB::bind_method(D_METHOD("apply_deformation"), &CurveNetDeformer3D::apply_deformation);
}

void CurveNetDeformer3D::set_source_path(const NodePath &p_path) {
	source_path = p_path;
}

NodePath CurveNetDeformer3D::get_source_path() const {
	return source_path;
}

void CurveNetDeformer3D::set_profile_curves(const TypedArray<Curve3D> &p_curves) {
	profile_curves = p_curves;
}

TypedArray<Curve3D> CurveNetDeformer3D::get_profile_curves() const {
	return profile_curves;
}

void CurveNetDeformer3D::set_length_tiebreak(double p_v) {
	length_tiebreak = p_v;
}

double CurveNetDeformer3D::get_length_tiebreak() const {
	return length_tiebreak;
}

void CurveNetDeformer3D::set_deformation_active(bool p_v) {
	deformation_active = p_v;
	if (deformation_active) {
		apply_deformation();
	}
}

bool CurveNetDeformer3D::is_deformation_active() const {
	return deformation_active;
}

void CurveNetDeformer3D::apply_deformation() {
	// STUB: Cycle 6 GREEN will:
	//   1. Pull triangles from the source MeshInstance3D's mesh.
	//   2. Run curvenet::tris_to_quads to fuse pairs into quads.
	//   3. Auto-bind each source vertex to (face, s, t).
	//   4. Build curvenet::CurveNet from profile_curves (Curve3D -> ProfileCurve).
	//   5. Call curvenet::deform and write back to a new ArrayMesh.
	// For now this is a no-op so the class registers and the demo loads.
	UtilityFunctions::print("CurveNetDeformer3D::apply_deformation() — TODO cycle 6 GREEN");
}

} // namespace godot
