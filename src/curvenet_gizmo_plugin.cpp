// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
#include "curvenet_gizmo_plugin.h"

#include "curvenet_deformer_3d.h"

#include "curvenet/curvenet_builder.h"
#include "curvenet/surface_projection.h"
#include "curvenet/vec3.h"

#include <godot_cpp/classes/array_mesh.hpp>
#include <godot_cpp/classes/curve3d.hpp>
#include <godot_cpp/classes/editor_undo_redo_manager.hpp>
#include <godot_cpp/classes/geometry3d.hpp>
#include <godot_cpp/classes/mesh.hpp>
#include <godot_cpp/classes/mesh_instance3d.hpp>
#include <godot_cpp/classes/standard_material3d.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/callable.hpp>
#include <godot_cpp/variant/color.hpp>
#include <godot_cpp/variant/packed_int32_array.hpp>
#include <godot_cpp/variant/packed_vector3_array.hpp>
#include <godot_cpp/variant/transform3d.hpp>
#include <godot_cpp/variant/typed_array.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#include <godot_cpp/variant/variant.hpp>

#include <vector>

namespace godot {

void CurveNetGizmoPlugin::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_redraw", "gizmo"), &CurveNetGizmoPlugin::_redraw);
}

CurveNetGizmoPlugin::CurveNetGizmoPlugin() {
	create_material("anchor",          Color(1.0f, 0.4f, 0.4f, 1.0f));
	create_material("regular",         Color(1.0f, 1.0f, 1.0f, 1.0f));
	create_material("intersection",    Color(0.4f, 1.0f, 0.4f, 1.0f));
	create_material("tangent",         Color(1.0f, 0.85f, 0.3f, 1.0f));
	create_material("projection_link", Color(0.4f, 0.8f, 1.0f, 1.0f));
	create_handle_material("knot_handles", false);
}

namespace {

// Rebuild the curvenet graph from the deformer's live profile_curves.
// The graph structure is deterministic (insertion order in build()) so
// _redraw, _set_handle, _commit_handle, _get_handle_value can all
// independently rebuild it and agree on which knot index corresponds
// to which handle id.
inline curvenet::curvenet_builder::CurvenetGraph
build_graph_from_deformer(CurveNetDeformer3D *deformer) {
	const TypedArray<Curve3D> curves = deformer->get_profile_curves();
	std::vector<curvenet::curvenet_builder::CurvePoints> input;
	input.reserve(curves.size());
	for (int i = 0; i < curves.size(); ++i) {
		Ref<Curve3D> c = curves[i];
		curvenet::curvenet_builder::CurvePoints pts;
		if (c.is_valid()) {
			const int n = c->get_point_count();
			pts.reserve(n);
			for (int j = 0; j < n; ++j) {
				Vector3 p = c->get_point_position(j);
				pts.push_back({ p.x, p.y, p.z });
			}
		}
		input.push_back(pts);
	}
	return curvenet::curvenet_builder::build(input, 1.0e-6);
}

} // namespace

void CurveNetGizmoPlugin::set_editor_plugin(EditorPlugin *p_plugin) {
	editor_plugin = p_plugin;
}

bool CurveNetGizmoPlugin::_has_gizmo(Node3D *p_node) const {
	return Object::cast_to<CurveNetDeformer3D>(p_node) != nullptr;
}

Ref<EditorNode3DGizmo> CurveNetGizmoPlugin::_create_gizmo(Node3D *p_node) const {
	if (!_has_gizmo(p_node)) {
		return Ref<EditorNode3DGizmo>();
	}
	Ref<EditorNode3DGizmo> gizmo;
	gizmo.instantiate();
	// Bind the gizmo so the slot signature matches `_redraw(Ref<...>)`.
	p_node->connect(
		"_curvenet_redraw_request",
		Callable(const_cast<CurveNetGizmoPlugin *>(this), "_redraw").bind(gizmo));
	// 4.6 doesn't auto-fire _redraw after _create_gizmo; defer one emit so
	// the gizmo paints once on first selection.
	p_node->call_deferred("emit_signal", "_curvenet_redraw_request");
	return gizmo;
}

String CurveNetGizmoPlugin::_get_gizmo_name() const {
	return "CurveNetGizmo";
}

String CurveNetGizmoPlugin::_get_handle_name(const Ref<EditorNode3DGizmo> &p_gizmo,
                                               int32_t p_id, bool /*p_secondary*/) const {
	return String("knot ") + String::num_int64(p_id);
}

Variant CurveNetGizmoPlugin::_get_handle_value(const Ref<EditorNode3DGizmo> &p_gizmo,
                                                  int32_t p_id, bool /*p_secondary*/) const {
	if (p_gizmo.is_null()) {
		return Variant();
	}
	CurveNetDeformer3D *deformer =
		Object::cast_to<CurveNetDeformer3D>(p_gizmo->get_node_3d());
	if (deformer == nullptr) {
		return Variant();
	}
	const auto graph = build_graph_from_deformer(deformer);
	if (p_id < 0 || static_cast<std::size_t>(p_id) >= graph.knot_positions.size()) {
		return Variant();
	}
	const curvenet::Vec3 &p = graph.knot_positions[p_id];
	return Vector3(static_cast<float>(p.x), static_cast<float>(p.y), static_cast<float>(p.z));
}

void CurveNetGizmoPlugin::_set_handle(const Ref<EditorNode3DGizmo> &p_gizmo,
                                        int32_t p_id, bool /*p_secondary*/,
                                        Camera3D *p_camera, const Vector2 &p_screen_pos) {
	if (p_gizmo.is_null() || p_camera == nullptr) {
		return;
	}
	CurveNetDeformer3D *deformer =
		Object::cast_to<CurveNetDeformer3D>(p_gizmo->get_node_3d());
	if (deformer == nullptr) {
		return;
	}
	const auto graph = build_graph_from_deformer(deformer);
	if (p_id < 0 || static_cast<std::size_t>(p_id) >= graph.knot_positions.size()) {
		return;
	}
	const Vector3 ray_from = p_camera->project_ray_origin(p_screen_pos);
	const Vector3 ray_dir  = p_camera->project_ray_normal(p_screen_pos);
	const curvenet::Vec3 &k = graph.knot_positions[p_id];
	const Vector3 plane_point(static_cast<float>(k.x),
	                            static_cast<float>(k.y),
	                            static_cast<float>(k.z));
	const Vector3 plane_normal = p_camera->get_camera_transform().basis.get_column(2);
	Plane plane(plane_normal, plane_point);
	Array planes;
	planes.append(plane);
	PackedVector3Array hits = Geometry3D::get_singleton()->segment_intersects_convex(
		ray_from, ray_from + ray_dir * 16384.0f, planes);
	if (hits.is_empty()) {
		return;
	}
	const Vector3 hit_world = hits[0];
	// Drag every (curve, handle) pair that ε-merged onto this knot.
	const TypedArray<Curve3D> curves = deformer->get_profile_curves();
	for (const curvenet::curvenet_builder::KnotRef &ref : graph.incidence[p_id]) {
		Ref<Curve3D> c = curves[ref.curve_id];
		if (c.is_valid() && ref.knot_idx >= 0 && ref.knot_idx < c->get_point_count()) {
			c->set_point_position(ref.knot_idx, hit_world);
		}
	}
	// Re-run the deformer so the gizmo + the source mesh both refresh.
	deformer->apply_deformation();
}

void CurveNetGizmoPlugin::_commit_handle(const Ref<EditorNode3DGizmo> &p_gizmo,
                                           int32_t p_id, bool /*p_secondary*/,
                                           const Variant &p_restore, bool p_cancel) {
	if (p_gizmo.is_null()) {
		return;
	}
	CurveNetDeformer3D *deformer =
		Object::cast_to<CurveNetDeformer3D>(p_gizmo->get_node_3d());
	if (deformer == nullptr) {
		return;
	}
	const TypedArray<Curve3D> curves = deformer->get_profile_curves();
	const auto graph = build_graph_from_deformer(deformer);
	if (p_id < 0 || static_cast<std::size_t>(p_id) >= graph.knot_positions.size()) {
		return;
	}

	if (p_cancel) {
		// User cancelled: restore every incident handle to the saved
		// position.
		const Vector3 old = p_restore;
		for (const curvenet::curvenet_builder::KnotRef &ref : graph.incidence[p_id]) {
			Ref<Curve3D> c = curves[ref.curve_id];
			if (c.is_valid() && ref.knot_idx >= 0 && ref.knot_idx < c->get_point_count()) {
				c->set_point_position(ref.knot_idx, old);
			}
		}
		deformer->apply_deformation();
		return;
	}

	if (editor_plugin == nullptr) {
		return;
	}
	EditorUndoRedoManager *undo = editor_plugin->get_undo_redo();
	if (undo == nullptr) {
		return;
	}

	const curvenet::Vec3 &k = graph.knot_positions[p_id];
	const Vector3 current(static_cast<float>(k.x),
	                       static_cast<float>(k.y),
	                       static_cast<float>(k.z));
	undo->create_action("Move curvenet knot " + String::num_int64(p_id));
	for (const curvenet::curvenet_builder::KnotRef &ref : graph.incidence[p_id]) {
		Ref<Curve3D> c = curves[ref.curve_id];
		if (c.is_null() || ref.knot_idx < 0 || ref.knot_idx >= c->get_point_count()) {
			continue;
		}
		undo->add_do_method(c.ptr(),   "set_point_position", ref.knot_idx, current);
		undo->add_undo_method(c.ptr(), "set_point_position", ref.knot_idx, p_restore);
	}
	undo->add_do_method(deformer,   "apply_deformation");
	undo->add_undo_method(deformer, "apply_deformation");
	undo->commit_action(false);
}

void CurveNetGizmoPlugin::_redraw(const Ref<EditorNode3DGizmo> &p_gizmo) {
	if (p_gizmo.is_null()) {
		return;
	}
	p_gizmo->clear();

	CurveNetDeformer3D *deformer =
		Object::cast_to<CurveNetDeformer3D>(p_gizmo->get_node_3d());
	if (deformer == nullptr) {
		return;
	}

	// Pull the rest knot positions straight from the live profile_curves
	// so the gizmo updates as the user drags Curve3D handles.
	const TypedArray<Curve3D> curves = deformer->get_profile_curves();
	std::vector<curvenet::curvenet_builder::CurvePoints> input_curves;
	input_curves.reserve(curves.size());
	for (int i = 0; i < curves.size(); ++i) {
		Ref<Curve3D> c = curves[i];
		curvenet::curvenet_builder::CurvePoints pts;
		if (c.is_valid()) {
			const int n = c->get_point_count();
			pts.reserve(n);
			for (int j = 0; j < n; ++j) {
				Vector3 p = c->get_point_position(j);
				pts.push_back({ p.x, p.y, p.z });
			}
		}
		input_curves.push_back(pts);
	}
	const auto graph = curvenet::curvenet_builder::build(input_curves, 1.0e-6);
	const auto kinds = curvenet::curvenet_builder::classify(graph);
	const auto tangents = curvenet::curvenet_builder::outgoing_tangents(graph);

	// Knot markers: small 3-axis "+" cross per knot, bucketed by kind so
	// each kind gets its own coloured material.
	const float marker_size = 0.04f;
	auto add_marker = [marker_size](PackedVector3Array &out, const Vector3 &p) {
		out.push_back(p + Vector3(-marker_size, 0, 0));
		out.push_back(p + Vector3( marker_size, 0, 0));
		out.push_back(p + Vector3(0, -marker_size, 0));
		out.push_back(p + Vector3(0,  marker_size, 0));
		out.push_back(p + Vector3(0, 0, -marker_size));
		out.push_back(p + Vector3(0, 0,  marker_size));
	};
	PackedVector3Array anchor_lines;
	PackedVector3Array regular_lines;
	PackedVector3Array intersection_lines;
	for (std::size_t k = 0; k < graph.knot_positions.size(); ++k) {
		const Vector3 p(graph.knot_positions[k].x,
		                 graph.knot_positions[k].y,
		                 graph.knot_positions[k].z);
		switch (kinds[k]) {
			case curvenet::curvenet_builder::KnotKind::anchor:
				add_marker(anchor_lines, p);
				break;
			case curvenet::curvenet_builder::KnotKind::intersection:
				add_marker(intersection_lines, p);
				break;
			case curvenet::curvenet_builder::KnotKind::regular:
			default:
				add_marker(regular_lines, p);
				break;
		}
	}
	if (anchor_lines.size() > 0) {
		p_gizmo->add_lines(
			anchor_lines, get_material("anchor", p_gizmo), false,
			Color(1.0f, 0.4f, 0.4f, 1.0f));
	}
	if (regular_lines.size() > 0) {
		p_gizmo->add_lines(
			regular_lines, get_material("regular", p_gizmo), false,
			Color(1.0f, 1.0f, 1.0f, 1.0f));
	}
	if (intersection_lines.size() > 0) {
		p_gizmo->add_lines(
			intersection_lines, get_material("intersection", p_gizmo), false,
			Color(0.4f, 1.0f, 0.4f, 1.0f));
	}

	// Draggable handles: one per merged knot, in knot-index order so
	// _set_handle / _commit_handle / _get_handle_value can rebuild the
	// graph deterministically and resolve `id` -> merged-knot index.
	PackedVector3Array handle_positions;
	PackedInt32Array   handle_ids;
	handle_positions.resize(graph.knot_positions.size());
	handle_ids.resize(graph.knot_positions.size());
	for (std::size_t k = 0; k < graph.knot_positions.size(); ++k) {
		handle_positions.set(k, Vector3(
			static_cast<float>(graph.knot_positions[k].x),
			static_cast<float>(graph.knot_positions[k].y),
			static_cast<float>(graph.knot_positions[k].z)));
		handle_ids.set(k, static_cast<int>(k));
	}
	if (handle_positions.size() > 0) {
		p_gizmo->add_handles(
			handle_positions, get_material("knot_handles", p_gizmo),
			handle_ids, false, false);
	}

	// Tangent rays at intersection knots: short line per outgoing tangent.
	const float tangent_ray_len = 0.15f;
	PackedVector3Array tangent_lines;
	for (std::size_t k = 0; k < graph.knot_positions.size(); ++k) {
		if (kinds[k] != curvenet::curvenet_builder::KnotKind::intersection) {
			continue;
		}
		const Vector3 origin(graph.knot_positions[k].x,
		                      graph.knot_positions[k].y,
		                      graph.knot_positions[k].z);
		for (const curvenet::Vec3 &t : tangents[k]) {
			tangent_lines.push_back(origin);
			tangent_lines.push_back(Vector3(
				origin.x + static_cast<float>(t.x) * tangent_ray_len,
				origin.y + static_cast<float>(t.y) * tangent_ray_len,
				origin.z + static_cast<float>(t.z) * tangent_ray_len));
		}
	}
	if (tangent_lines.size() > 0) {
		p_gizmo->add_lines(
			tangent_lines, get_material("tangent", p_gizmo), false,
			Color(1.0f, 0.85f, 0.3f, 1.0f));
	}

	// Projection links: from each merged knot to its closest mesh vertex
	// on the source MeshInstance3D. Re-runs the projection (cheap for
	// small curvenets, deferred to the deformer's cache for production).
	const NodePath src_path = deformer->get_source_path();
	Node *src_node = deformer->get_node_or_null(src_path);
	MeshInstance3D *src = Object::cast_to<MeshInstance3D>(src_node);
	if (src != nullptr) {
		Ref<Mesh> mesh = src->get_mesh();
		if (mesh.is_valid()) {
			std::vector<curvenet::Vec3> mesh_positions;
			for (int s = 0; s < mesh->get_surface_count(); ++s) {
				Array arrays = mesh->surface_get_arrays(s);
				PackedVector3Array verts = arrays[Mesh::ARRAY_VERTEX];
				for (int i = 0; i < verts.size(); ++i) {
					Vector3 v = verts[i];
					mesh_positions.push_back({ v.x, v.y, v.z });
				}
			}
			const auto projections =
				curvenet::surface_projection::project_to_vertices(
					graph.knot_positions, mesh_positions);
			PackedVector3Array link_lines;
			for (std::size_t i = 0; i < projections.size(); ++i) {
				const Vector3 from(graph.knot_positions[i].x,
				                    graph.knot_positions[i].y,
				                    graph.knot_positions[i].z);
				const Vector3 to(projections[i].position.x,
				                  projections[i].position.y,
				                  projections[i].position.z);
				link_lines.push_back(from);
				link_lines.push_back(to);
			}
			if (link_lines.size() > 0) {
				p_gizmo->add_lines(
					link_lines, get_material("projection_link", p_gizmo), false,
					Color(0.4f, 0.8f, 1.0f, 1.0f));
			}
		}
	}
}

} // namespace godot
