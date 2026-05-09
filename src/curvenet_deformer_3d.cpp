// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
#include "curvenet_deformer_3d.h"

#include "curvenet/tris_to_quads.h"

#include <godot_cpp/classes/array_mesh.hpp>
#include <godot_cpp/classes/mesh.hpp>
#include <godot_cpp/classes/mesh_instance3d.hpp>
#include <godot_cpp/classes/surface_tool.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/packed_int32_array.hpp>
#include <godot_cpp/variant/packed_vector3_array.hpp>
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
	// Cycle 6 GREEN — first slice:
	//   - Pull triangles from the source MeshInstance3D.
	//   - Run curvenet::tris_to_quads to fuse pairs into quads.
	//   - Rebuild a quad-dominant ArrayMesh with the SurfaceTool.
	//
	// Profile-curve binding + Coons-patch deformation will be wired in the
	// next iteration; this cycle proves the full pipeline (mesh extraction →
	// tri-to-quad → mesh emit) compiles and round-trips.
	Node *src_node = get_node_or_null(source_path);
	MeshInstance3D *src = Object::cast_to<MeshInstance3D>(src_node);
	if (src == nullptr) {
		UtilityFunctions::printerr("CurveNetDeformer3D: source_path does not point to a MeshInstance3D");
		return;
	}
	Ref<Mesh> mesh = src->get_mesh();
	if (mesh.is_null()) {
		return;
	}

	curvenet::TriMesh tri;
	for (int s = 0; s < mesh->get_surface_count(); ++s) {
		Array arrays = mesh->surface_get_arrays(s);
		PackedVector3Array verts = arrays[Mesh::ARRAY_VERTEX];
		PackedInt32Array indices = arrays[Mesh::ARRAY_INDEX];
		const int v_offset = static_cast<int>(tri.vertices.size());
		for (int i = 0; i < verts.size(); ++i) {
			Vector3 v = verts[i];
			tri.vertices.push_back({ v.x, v.y, v.z });
		}
		// If indices are present use them; otherwise tris are sequential.
		if (indices.size() > 0) {
			for (int i = 0; i + 2 < indices.size(); i += 3) {
				tri.triangles.push_back({ v_offset + indices[i],
						v_offset + indices[i + 1],
						v_offset + indices[i + 2] });
			}
		} else {
			for (int i = 0; i + 2 < verts.size(); i += 3) {
				tri.triangles.push_back({ v_offset + i, v_offset + i + 1, v_offset + i + 2 });
			}
		}
	}

	curvenet::TrisToQuadsParams params;
	params.length_tiebreak = length_tiebreak;
	curvenet::PolyMesh poly = curvenet::tris_to_quads(tri, params);

	// Re-emit as a triangle ArrayMesh (quads triangulated for Godot rendering;
	// the quad topology is preserved internally for patch evaluation in the
	// next slice).
	Ref<SurfaceTool> st;
	st.instantiate();
	st->begin(Mesh::PRIMITIVE_TRIANGLES);
	for (const auto &v : poly.vertices) {
		st->add_vertex(Vector3(v.x, v.y, v.z));
	}
	for (const auto &f : poly.faces) {
		if (f.count == 3) {
			st->add_index(f.v[0]);
			st->add_index(f.v[1]);
			st->add_index(f.v[2]);
		} else if (f.count == 4) {
			// Triangulate the quad along the (v0, v2) diagonal.
			st->add_index(f.v[0]);
			st->add_index(f.v[1]);
			st->add_index(f.v[2]);
			st->add_index(f.v[0]);
			st->add_index(f.v[2]);
			st->add_index(f.v[3]);
		}
	}
	st->generate_normals();
	Ref<ArrayMesh> out_mesh = st->commit();
	set_mesh(out_mesh);

	UtilityFunctions::print("CurveNetDeformer3D: ", static_cast<int>(tri.triangles.size()),
			" tris -> ", static_cast<int>(poly.faces.size()), " faces");
}

} // namespace godot
