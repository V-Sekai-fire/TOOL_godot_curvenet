// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
#include "vertex_handles_gizmo_plugin.h"

#include "vertex_handles_3d.h"

#include <godot_cpp/classes/editor_undo_redo_manager.hpp>
#include <godot_cpp/classes/geometry3d.hpp>
#include <godot_cpp/classes/mesh.hpp>
#include <godot_cpp/classes/mesh_data_tool.hpp>
#include <godot_cpp/classes/mesh_instance3d.hpp>
#include <godot_cpp/classes/standard_material3d.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/callable.hpp>
#include <godot_cpp/variant/packed_vector3_array.hpp>
#include <godot_cpp/variant/plane.hpp>
#include <godot_cpp/variant/transform3d.hpp>

namespace godot {

void VertexHandlesGizmoPlugin::_bind_methods() {
	// Internal callable wired from VertexHandles3D::_request_redraw -> _redraw(gizmo).
	ClassDB::bind_method(D_METHOD("_redraw", "gizmo"), &VertexHandlesGizmoPlugin::_redraw);
}

VertexHandlesGizmoPlugin::VertexHandlesGizmoPlugin() {
	create_material("main", Color(1.0f, 1.0f, 1.0f, 1.0f), false, true, true);
	create_handle_material("handles", false);
}

void VertexHandlesGizmoPlugin::set_editor_plugin(EditorPlugin *p_plugin) {
	editor_plugin = p_plugin;
}

bool VertexHandlesGizmoPlugin::_has_gizmo(Node3D *p_node) const {
	return Object::cast_to<VertexHandles3D>(p_node) != nullptr;
}

Ref<EditorNode3DGizmo> VertexHandlesGizmoPlugin::_create_gizmo(Node3D *p_node) const {
	if (!_has_gizmo(p_node)) {
		return Ref<EditorNode3DGizmo>();
	}
	Ref<EditorNode3DGizmo> gizmo;
	gizmo.instantiate();
	// Allow the node to request a gizmo redraw when its state changes. Bind the
	// gizmo so the slot signature matches `_redraw(Ref<EditorNode3DGizmo>)`.
	p_node->connect(
			"_request_redraw",
			Callable(const_cast<VertexHandlesGizmoPlugin *>(this), "_redraw").bind(gizmo));
	return gizmo;
}

String VertexHandlesGizmoPlugin::_get_gizmo_name() const {
	return "VertexHandlesGizmo";
}

String VertexHandlesGizmoPlugin::_get_handle_name(const Ref<EditorNode3DGizmo> &p_gizmo, int32_t p_id, bool p_secondary) const {
	return String::num_int64(p_id);
}

Variant VertexHandlesGizmoPlugin::_get_handle_value(const Ref<EditorNode3DGizmo> &p_gizmo, int32_t p_id, bool p_secondary) const {
	if (p_gizmo.is_null()) {
		return Variant();
	}
	VertexHandles3D *vh = Object::cast_to<VertexHandles3D>(p_gizmo->get_node_3d());
	if (vh == nullptr) {
		return Variant();
	}
	TypedArray<PackedVector3Array> arrays = vh->get_point_arrays();
	if (arrays.is_empty()) {
		return Variant();
	}
	PackedVector3Array first = arrays[0];
	if (p_id < 0 || p_id >= first.size()) {
		return Variant();
	}
	return first[p_id];
}

void VertexHandlesGizmoPlugin::_set_handle(const Ref<EditorNode3DGizmo> &p_gizmo, int32_t p_id, bool p_secondary, Camera3D *p_camera, const Vector2 &p_screen_pos) {
	if (p_gizmo.is_null() || p_camera == nullptr) {
		return;
	}
	VertexHandles3D *vh = Object::cast_to<VertexHandles3D>(p_gizmo->get_node_3d());
	if (vh == nullptr) {
		return;
	}
	TypedArray<PackedVector3Array> arrays = vh->get_point_arrays();
	if (arrays.is_empty()) {
		return;
	}
	PackedVector3Array first = arrays[0];
	if (p_id < 0 || p_id >= first.size()) {
		return;
	}

	const Vector3 ray_from = p_camera->project_ray_origin(p_screen_pos);
	const Vector3 ray_dir = p_camera->project_ray_normal(p_screen_pos);

	const Transform3D xform = vh->get_global_transform();
	const Vector3 plane_point = xform.xform(first[p_id]);
	const Vector3 plane_normal = p_camera->get_camera_transform().basis.get_column(2);
	Plane plane(plane_normal, plane_point);

	Array planes;
	planes.append(plane);
	PackedVector3Array hits = Geometry3D::get_singleton()->segment_intersects_convex(
			ray_from, ray_from + ray_dir * 16384.0f, planes);
	if (hits.is_empty()) {
		return;
	}

	const Vector3 hit_local = xform.affine_inverse().xform(hits[0]);
	vh->set_point(0, p_id, hit_local);

	_redraw(p_gizmo);
}

void VertexHandlesGizmoPlugin::_commit_handle(const Ref<EditorNode3DGizmo> &p_gizmo, int32_t p_id, bool p_secondary, const Variant &p_restore, bool p_cancel) {
	if (p_gizmo.is_null() || editor_plugin == nullptr) {
		return;
	}
	VertexHandles3D *vh = Object::cast_to<VertexHandles3D>(p_gizmo->get_node_3d());
	if (vh == nullptr) {
		return;
	}
	TypedArray<PackedVector3Array> arrays = vh->get_point_arrays();
	if (arrays.is_empty()) {
		return;
	}
	PackedVector3Array first = arrays[0];
	if (p_id < 0 || p_id >= first.size()) {
		return;
	}

	EditorUndoRedoManager *undo = editor_plugin->get_undo_redo();
	if (undo == nullptr) {
		return;
	}
	undo->create_action("Move handle " + String::num_int64(p_id));
	undo->add_do_method(vh, "set_point", 0, p_id, first[p_id]);
	undo->add_undo_method(vh, "set_point", 0, p_id, p_restore);
	undo->commit_action(false);
}

void VertexHandlesGizmoPlugin::_redraw(const Ref<EditorNode3DGizmo> &p_gizmo) {
	if (p_gizmo.is_null()) {
		return;
	}
	p_gizmo->clear();

	VertexHandles3D *vh = Object::cast_to<VertexHandles3D>(p_gizmo->get_node_3d());
	if (vh == nullptr) {
		return;
	}
	MeshInstance3D *parent = Object::cast_to<MeshInstance3D>(vh->get_parent());
	if (parent == nullptr) {
		return;
	}
	Ref<Mesh> mesh = parent->get_mesh();

	if (vh->get_wireframe() && mesh.is_valid()) {
		PackedVector3Array lines;
		Ref<MeshDataTool> mdt;
		mdt.instantiate();
		for (int sid = 0; sid < mesh->get_surface_count(); ++sid) {
			Ref<ArrayMesh> array_mesh = mesh;
			if (array_mesh.is_null()) {
				continue;
			}
			mdt->create_from_surface(array_mesh, sid);
			for (int fid = 0; fid < mdt->get_face_count(); ++fid) {
				for (int j = 0; j < 3; ++j) {
					lines.push_back(mdt->get_vertex(mdt->get_face_vertex(fid, j)));
					lines.push_back(mdt->get_vertex(mdt->get_face_vertex(fid, (j + 1) % 3)));
				}
			}
		}
		p_gizmo->add_lines(lines, get_material("main", p_gizmo), false, vh->get_wireframe_color());
	}

	PackedVector3Array handles;
	TypedArray<PackedVector3Array> arrays = vh->get_point_arrays();
	for (int i = 0; i < arrays.size(); ++i) {
		PackedVector3Array a = arrays[i];
		for (int j = 0; j < a.size(); ++j) {
			handles.push_back(a[j]);
		}
	}
	p_gizmo->add_handles(handles, get_material("handles", p_gizmo), PackedInt32Array(), false, false);
}

} // namespace godot
