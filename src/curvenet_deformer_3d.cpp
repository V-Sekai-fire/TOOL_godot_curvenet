// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
#include "curvenet_deformer_3d.h"

#include "curvenet/binding.h"
#include "curvenet/curvenet.h"
#include "curvenet/profile_curve.h"
#include "curvenet/tris_to_quads.h"

#include <godot_cpp/classes/array_mesh.hpp>
#include <godot_cpp/classes/curve3d.hpp>
#include <godot_cpp/classes/mesh.hpp>
#include <godot_cpp/classes/mesh_instance3d.hpp>
#include <godot_cpp/classes/surface_tool.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/packed_int32_array.hpp>
#include <godot_cpp/variant/packed_vector3_array.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

namespace godot {

// Local alias to disambiguate from godot::Vector3 in the tri-mesh push_back below.
using Vec3_t = curvenet::Vec3;

namespace {

// Convert a Godot Curve3D (point_position + relative point_in/point_out)
// into our pure-math closed-loop ProfileCurve.
curvenet::ProfileCurve to_profile_curve(const Ref<Curve3D> &c) {
	std::vector<curvenet::CurveHandle> handles;
	if (c.is_null()) {
		return curvenet::ProfileCurve{};
	}
	const int n = c->get_point_count();
	handles.reserve(n);
	for (int i = 0; i < n; ++i) {
		Vector3 p = c->get_point_position(i);
		Vector3 in_off = c->get_point_in(i);
		Vector3 out_off = c->get_point_out(i);
		handles.push_back({ { p.x, p.y, p.z },
				{ in_off.x, in_off.y, in_off.z },
				{ out_off.x, out_off.y, out_off.z } });
	}
	return curvenet::profile_from_handles(handles);
}

} // namespace

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
	ClassDB::bind_method(D_METHOD("get_face_count"), &CurveNetDeformer3D::get_face_count);
	ClassDB::bind_method(D_METHOD("get_face_vertex_count", "face_index"), &CurveNetDeformer3D::get_face_vertex_count);
	ClassDB::bind_method(D_METHOD("evaluate_face", "face_index", "s", "t"), &CurveNetDeformer3D::evaluate_face);
}

void CurveNetDeformer3D::invalidate_cache() {
	rest_cache.valid = false;
	rest_cache.poly = curvenet::PolyMesh{};
	rest_cache.bindings.clear();
	rest_cache.source_hash = 0;
}

void CurveNetDeformer3D::set_source_path(const NodePath &p_path) {
	source_path = p_path;
	invalidate_cache();
}

NodePath CurveNetDeformer3D::get_source_path() const {
	return source_path;
}

void CurveNetDeformer3D::set_profile_curves(const TypedArray<Curve3D> &p_curves) {
	profile_curves = p_curves;
	// Profile curves don't affect the rest-pose cache (only bindings do); no
	// invalidation needed.
}

TypedArray<Curve3D> CurveNetDeformer3D::get_profile_curves() const {
	return profile_curves;
}

void CurveNetDeformer3D::set_length_tiebreak(double p_v) {
	length_tiebreak = p_v;
	invalidate_cache(); // tri-to-quad weights depend on length_tiebreak
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

	// Hash the source mesh's vertex+index data so we can detect edits without
	// re-running tri->quad each call. RID is stable per Mesh resource so we
	// fold it together with the surface count and per-surface array sizes.
	uint64_t hash_val = static_cast<uint64_t>(mesh->get_rid().get_id());
	hash_val = hash_val * 1099511628211ULL + static_cast<uint64_t>(mesh->get_surface_count());
	for (int s = 0; s < mesh->get_surface_count(); ++s) {
		Array arrays = mesh->surface_get_arrays(s);
		PackedVector3Array verts = arrays[Mesh::ARRAY_VERTEX];
		PackedInt32Array indices = arrays[Mesh::ARRAY_INDEX];
		hash_val = hash_val * 1099511628211ULL + static_cast<uint64_t>(verts.size());
		hash_val = hash_val * 1099511628211ULL + static_cast<uint64_t>(indices.size());
	}

	if (rest_cache.valid && rest_cache.source_hash == hash_val) {
		// Rest-pose pipeline (tri->quad + bind) is cached; skip ahead to the
		// build-curves-and-deform path.
		goto rest_cache_hit;
	}

	{
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
	rest_cache.poly = curvenet::tris_to_quads(tri, params);
	rest_cache.bindings = curvenet::bind_polymesh(rest_cache.poly);
	rest_cache.source_hash = hash_val;
	rest_cache.valid = true;
	}

rest_cache_hit:
	const curvenet::PolyMesh &poly = rest_cache.poly;
	const std::vector<curvenet::BoundVertex> &bindings = rest_cache.bindings;

	// Build a CurveNet from authored Curve3D profile_curves. For now each
	// quad face's 4 boundaries come from straight-line bezier segments
	// between its corners, so deform with no curves authored is identity.
	// When profile_curves has 4 entries, we treat them as the four edges of
	// the (single) face and bind them in order.
	curvenet::CurveNet net;
	net.faces.reserve(poly.faces.size());
	std::vector<curvenet::ProfileCurve> profiles;
	profiles.reserve(profile_curves.size());
	for (int i = 0; i < profile_curves.size(); ++i) {
		Ref<Curve3D> c = profile_curves[i];
		profiles.push_back(to_profile_curve(c));
	}
	auto straight_curve_between = [](const Vec3_t &a, const Vec3_t &b) {
		curvenet::BoundaryCurve bc;
		bc.c0 = a;
		bc.c3 = b;
		bc.c1 = a + (b - a) * (1.0 / 3.0);
		bc.c2 = a + (b - a) * (2.0 / 3.0);
		return bc;
	};
	for (const auto &f : poly.faces) {
		if (f.count != 4) {
			net.faces.push_back(curvenet::BoundFace{});
			continue;
		}
		curvenet::BoundFace bf;
		const auto &v00 = poly.vertices[f.v[0]];
		const auto &v10 = poly.vertices[f.v[1]];
		const auto &v11 = poly.vertices[f.v[2]];
		const auto &v01 = poly.vertices[f.v[3]];
		// CCW boundary loop matching CoonsPatch from NgonPatch:
		//   bottom (P00->P10), right (P10->P11), top reversed (P11->P01), left reversed (P01->P00).
		bf.boundaries.push_back(straight_curve_between(v00, v10));
		bf.boundaries.push_back(straight_curve_between(v10, v11));
		bf.boundaries.push_back(straight_curve_between(v11, v01));
		bf.boundaries.push_back(straight_curve_between(v01, v00));
		// If the user authored exactly one Curve3D, use it for the bottom edge
		// of every face — quick way to see deformation in the demo. Future
		// slice: per-face-per-edge curve assignment.
		if (profiles.size() == 1) {
			const auto &p = profiles[0];
			if (p.handles.size() >= 2) {
				bf.boundaries[0].c0 = p.handles[0];
				bf.boundaries[0].c1 = p.tangents_out[0];
				bf.boundaries[0].c2 = p.tangents_in[1 % p.handles.size()];
				bf.boundaries[0].c3 = p.handles[1 % p.handles.size()];
			}
		}
		net.faces.push_back(bf);
	}

	// Deform using the bindings + curvenet.
	std::vector<curvenet::Vec3> deformed = curvenet::deform(net, bindings);

	UtilityFunctions::print("CurveNetDeformer3D: ", static_cast<int>(profiles.size()),
			" profiles, ", static_cast<int>(poly.faces.size()), " faces, ",
			static_cast<int>(deformed.size()), " deformed verts");

	// Re-emit as a triangle ArrayMesh (quads triangulated for Godot rendering).
	Ref<SurfaceTool> st;
	st.instantiate();
	st->begin(Mesh::PRIMITIVE_TRIANGLES);
	for (std::size_t i = 0; i < poly.vertices.size(); ++i) {
		const auto &v = (bindings[i].face_index >= 0) ? deformed[i] : poly.vertices[i];
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

	UtilityFunctions::print("CurveNetDeformer3D: ",
			static_cast<int>(poly.vertices.size()), " verts, ",
			static_cast<int>(poly.faces.size()), " faces (cache ",
			(rest_cache.valid ? "valid" : "miss"), ")");
}

int CurveNetDeformer3D::get_face_count() const {
	return rest_cache.valid ? static_cast<int>(rest_cache.poly.faces.size()) : 0;
}

int CurveNetDeformer3D::get_face_vertex_count(int face_index) const {
	if (!rest_cache.valid || face_index < 0 ||
			face_index >= static_cast<int>(rest_cache.poly.faces.size())) {
		return 0;
	}
	return rest_cache.poly.faces[face_index].count;
}

Vector3 CurveNetDeformer3D::evaluate_face(int face_index, double s, double t) {
	if (!rest_cache.valid) {
		// Build the cache lazily so the first call works without a prior apply_deformation().
		apply_deformation();
	}
	if (!rest_cache.valid || face_index < 0 ||
			face_index >= static_cast<int>(rest_cache.poly.faces.size())) {
		UtilityFunctions::printerr("CurveNetDeformer3D::evaluate_face: face_index out of range");
		return Vector3{ 0.0f, 0.0f, 0.0f };
	}
	const curvenet::PolyFace &f = rest_cache.poly.faces[face_index];
	if (f.count != 4) {
		// Triangle path: fall back to bilinear-on-corners (degenerate Coons handled in NgonPatch).
		// For now return the centroid as a stand-in.
		curvenet::Vec3 acc{ 0.0, 0.0, 0.0 };
		for (int i = 0; i < f.count; ++i) {
			acc += rest_cache.poly.vertices[f.v[i]];
		}
		acc = acc * (1.0 / f.count);
		return Vector3(acc.x, acc.y, acc.z);
	}
	auto straight = [](const curvenet::Vec3 &a, const curvenet::Vec3 &b) {
		curvenet::BoundaryCurve bc;
		bc.c0 = a;
		bc.c3 = b;
		bc.c1 = a + (b - a) * (1.0 / 3.0);
		bc.c2 = a + (b - a) * (2.0 / 3.0);
		return bc;
	};
	curvenet::CoonsPatch c;
	const auto &v00 = rest_cache.poly.vertices[f.v[0]];
	const auto &v10 = rest_cache.poly.vertices[f.v[1]];
	const auto &v11 = rest_cache.poly.vertices[f.v[2]];
	const auto &v01 = rest_cache.poly.vertices[f.v[3]];
	c.u0 = straight(v00, v10);
	c.v1 = straight(v10, v11);
	c.u1 = straight(v01, v11);
	c.v0 = straight(v00, v01);
	curvenet::Vec3 r = c.evaluate(s, t);
	return Vector3(r.x, r.y, r.z);
}

} // namespace godot
