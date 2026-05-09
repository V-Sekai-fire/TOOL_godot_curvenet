// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
#include "curvenet_gizmo_plugin.h"

#include "curvenet_deformer_3d.h"

#include "curvenet/curvenet_builder.h"
#include "curvenet/surface_projection.h"
#include "curvenet/vec3.h"

#include <godot_cpp/classes/array_mesh.hpp>
#include <godot_cpp/classes/curve3d.hpp>
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
}

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
