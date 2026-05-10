// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
#include "vertex_handles_3d.h"

#ifdef TOOLS_ENABLED
#include <godot_cpp/classes/editor_interface.hpp>
#include <godot_cpp/classes/editor_selection.hpp>
#endif
#include <godot_cpp/classes/engine.hpp>
#include <godot_cpp/classes/rendering_server.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/callable.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

namespace godot {

void VertexHandles3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_wireframe", "v"), &VertexHandles3D::set_wireframe);
	ClassDB::bind_method(D_METHOD("get_wireframe"), &VertexHandles3D::get_wireframe);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "wireframe"), "set_wireframe", "get_wireframe");

	ClassDB::bind_method(D_METHOD("set_wireframe_color", "c"), &VertexHandles3D::set_wireframe_color);
	ClassDB::bind_method(D_METHOD("get_wireframe_color"), &VertexHandles3D::get_wireframe_color);
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "wireframe_color"), "set_wireframe_color", "get_wireframe_color");

	ClassDB::bind_method(D_METHOD("set_point_arrays", "arrays"), &VertexHandles3D::set_point_arrays);
	ClassDB::bind_method(D_METHOD("get_point_arrays"), &VertexHandles3D::get_point_arrays);
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "point_arrays"), "set_point_arrays", "get_point_arrays");

	ClassDB::bind_method(D_METHOD("set_point", "surface", "point_idx", "p"), &VertexHandles3D::set_point);
	ClassDB::bind_method(D_METHOD("refresh_point_arrays"), &VertexHandles3D::refresh_point_arrays);
	ClassDB::bind_method(D_METHOD("update_mesh"), &VertexHandles3D::update_mesh);

	ADD_SIGNAL(MethodInfo("_request_redraw"));
}

void VertexHandles3D::set_wireframe(bool p_v) {
	wireframe = p_v;
	if (is_inside_tree()) {
		emit_signal("_request_redraw");
	}
}

bool VertexHandles3D::get_wireframe() const {
	return wireframe;
}

void VertexHandles3D::set_wireframe_color(const Color &p_c) {
	wireframe_color = p_c;
	if (is_inside_tree()) {
		emit_signal("_request_redraw");
	}
}

Color VertexHandles3D::get_wireframe_color() const {
	return wireframe_color;
}

void VertexHandles3D::set_point_arrays(const TypedArray<PackedVector3Array> &p_arrays) {
	point_arrays = p_arrays;
	if (is_inside_tree()) {
		update_mesh();
	}
}

TypedArray<PackedVector3Array> VertexHandles3D::get_point_arrays() const {
	return point_arrays;
}

void VertexHandles3D::_ready() {
	refresh_point_arrays();

#ifdef TOOLS_ENABLED
	// Editor-only: refresh on selection change. EditorInterface only exists
	// when running inside the editor — both compile-time (template builds
	// drop the headers) and runtime (`is_editor_hint()` returns false).
	if (Engine::get_singleton()->is_editor_hint()) {
		EditorInterface *ei = EditorInterface::get_singleton();
		if (ei != nullptr) {
			EditorSelection *sel = ei->get_selection();
			if (sel != nullptr) {
				sel->connect("selection_changed", Callable(this, "refresh_point_arrays"));
			}
		}
	}
#endif

	// Wire to a sibling CurveNetDeformer3D so handle moves trigger redeform.
	Node *deformer = find_curvenet_deformer();
	if (deformer != nullptr) {
		connect("_request_redraw", Callable(deformer, "apply_deformation"));
	}
}

void VertexHandles3D::set_point(int p_surface, int p_point_idx, const Vector3 &p_p) {
	if (p_surface < 0 || p_surface >= point_arrays.size()) {
		return;
	}
	PackedVector3Array arr = point_arrays[p_surface];
	if (p_point_idx < 0 || p_point_idx >= arr.size()) {
		return;
	}
	arr.set(p_point_idx, p_p);
	point_arrays[p_surface] = arr;
	update_mesh();
}

void VertexHandles3D::refresh_point_arrays() {
#ifdef TOOLS_ENABLED
	// Only refresh in the editor when this node is selected (matches GDScript
	// semantics so we don't clobber edits made elsewhere). Template builds
	// always refresh — there's no editor selection state to consult.
	if (Engine::get_singleton()->is_editor_hint()) {
		EditorInterface *ei = EditorInterface::get_singleton();
		if (ei == nullptr) {
			return;
		}
		EditorSelection *sel = ei->get_selection();
		if (sel == nullptr) {
			return;
		}
		bool selected = false;
		TypedArray<Node> nodes = sel->get_selected_nodes();
		for (int i = 0; i < nodes.size(); ++i) {
			Node *n = Object::cast_to<Node>(nodes[i]);
			if (n == this) {
				selected = true;
				break;
			}
		}
		if (!selected) {
			return;
		}
	}
#endif

	Ref<Mesh> current = get_mesh();
	if (current.is_null()) {
		return;
	}
	// Re-host as ArrayMesh so we can call clear_surfaces / add_surface_from_arrays.
	Ref<ArrayMesh> array_mesh = to_array_mesh(current);
	set_mesh(array_mesh);

	point_arrays.clear();
	for (int i = 0; i < array_mesh->get_surface_count(); ++i) {
		Array arrays = array_mesh->surface_get_arrays(i);
		PackedVector3Array verts = arrays[Mesh::ARRAY_VERTEX];
		point_arrays.push_back(verts);
	}
}

void VertexHandles3D::update_mesh() {
	if (!is_node_ready() || point_arrays.is_empty()) {
		return;
	}
	Ref<ArrayMesh> array_mesh = get_mesh();
	if (array_mesh.is_null()) {
		return;
	}

	std::vector<Array> surface_arrays;
	surface_arrays.reserve(array_mesh->get_surface_count());
	for (int i = 0; i < array_mesh->get_surface_count(); ++i) {
		surface_arrays.push_back(array_mesh->surface_get_arrays(i));
	}

	const Mesh::PrimitiveType prim = primitive_type_of(array_mesh);

	array_mesh->clear_surfaces();

	for (int i = 0; i < static_cast<int>(surface_arrays.size()); ++i) {
		Array arrays = surface_arrays[i];
		PackedVector3Array verts = point_arrays[i];
		arrays[Mesh::ARRAY_VERTEX] = verts;
		array_mesh->add_surface_from_arrays(prim, arrays);
	}

	emit_signal("_request_redraw");
}

Node *VertexHandles3D::find_curvenet_deformer() const {
	const StringName klass = StringName("CurveNetDeformer3D");
	Node *parent = get_parent();
	if (parent == nullptr) {
		return nullptr;
	}
	for (int i = 0; i < parent->get_child_count(); ++i) {
		Node *c = parent->get_child(i);
		if (c == this || c == nullptr) {
			continue;
		}
		if (c->get_class() == klass) {
			return c;
		}
	}
	return nullptr;
}

Ref<ArrayMesh> VertexHandles3D::to_array_mesh(const Ref<Mesh> &p_mesh) const {
	std::vector<Array> surface_arrays;
	surface_arrays.reserve(p_mesh->get_surface_count());
	for (int i = 0; i < p_mesh->get_surface_count(); ++i) {
		surface_arrays.push_back(p_mesh->surface_get_arrays(i));
	}
	const Mesh::PrimitiveType prim = primitive_type_of(p_mesh);

	Ref<ArrayMesh> out;
	out.instantiate();
	for (const Array &arr : surface_arrays) {
		out->add_surface_from_arrays(prim, arr);
	}
	return out;
}

Mesh::PrimitiveType VertexHandles3D::primitive_type_of(const Ref<Mesh> &p_mesh) const {
	if (p_mesh.is_null() || p_mesh->get_surface_count() == 0) {
		return Mesh::PRIMITIVE_TRIANGLES;
	}
	Dictionary info = RenderingServer::get_singleton()->mesh_get_surface(p_mesh->get_rid(), 0);
	return static_cast<Mesh::PrimitiveType>(static_cast<int>(info.get("primitive", Mesh::PRIMITIVE_TRIANGLES)));
}

} // namespace godot
