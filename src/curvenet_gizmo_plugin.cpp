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
	create_material("tangent_link",    Color(1.0f, 0.85f, 0.3f, 0.6f));
	create_material("projection_link", Color(0.4f, 0.8f, 1.0f, 1.0f));
	// DeGoes22 §3 per-knot frame axes (red = tangent, green = normal,
	// blue = binormal — matches Blender/Godot bone axis convention).
	create_material("frame_t", Color(1.0f, 0.3f, 0.3f, 1.0f));
	create_material("frame_n", Color(0.3f, 1.0f, 0.3f, 1.0f));
	create_material("frame_b", Color(0.3f, 0.5f, 1.0f, 1.0f));
	create_material("width_ring", Color(0.3f, 1.0f, 0.7f, 0.6f));
	create_handle_material("knot_handles",     false);
	create_handle_material("tangent_handles",  false);
	create_handle_material("tilt_handles",     false);
	create_handle_material("width_handles",    false);
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

// Tangent-handle reference: which Curve3D handle and which side
// (point_in vs point_out).
struct TangentRef {
	int  curve_id;
	int  knot_idx;
	bool is_out;
};

// (curve, knot) reference for tilt + width drag handles.
struct KnotPosRef {
	int curve_id;
	int knot_idx;
};

// Flat enumeration of every (curve, knot) pair, in deterministic order so
// the gizmo's id ↔ ref mapping is stable across redraw / set / commit.
inline std::vector<KnotPosRef>
enumerate_all_knots(const TypedArray<Curve3D> &curves) {
	std::vector<KnotPosRef> out;
	for (int c = 0; c < curves.size(); ++c) {
		Ref<Curve3D> cv = curves[c];
		if (cv.is_null()) continue;
		const int n = cv->get_point_count();
		for (int k = 0; k < n; ++k) {
			out.push_back({ c, k });
		}
	}
	return out;
}

// Per-knot DeGoes22 §3 frame: tangent + un-tilted reference (b0, n0)
// derived from world-up Gram-Schmidt + tilt-rotated (b, n). Reused by
// both the visualization (_redraw) and the drag-handle math
// (_set_handle / _get_handle_value).
struct KnotFrame {
	Vector3 origin;
	Vector3 t;
	Vector3 b0, n0;  // un-tilted reference
	Vector3 b,  n;   // tilt-rotated
	float   tilt;
	bool    valid;
};

inline KnotFrame compute_knot_frame(const Ref<Curve3D> &c, int ki) {
	KnotFrame kf;
	kf.valid = false;
	if (c.is_null()) return kf;
	const int n = c->get_point_count();
	if (ki < 0 || ki >= n) return kf;
	kf.origin = c->get_point_position(ki);
	Vector3 t_dir;
	if (ki + 1 < n) {
		t_dir = c->get_point_position(ki + 1) - kf.origin;
	} else if (ki - 1 >= 0) {
		t_dir = kf.origin - c->get_point_position(ki - 1);
	} else {
		t_dir = Vector3(1.0f, 0.0f, 0.0f);
	}
	const float tlen = t_dir.length();
	if (tlen < 1e-6f) return kf;
	kf.t = t_dir / tlen;
	Vector3 ref_up(0.0f, 1.0f, 0.0f);
	if (std::fabs(kf.t.dot(ref_up)) > 0.95f) {
		ref_up = Vector3(0.0f, 0.0f, 1.0f);
	}
	Vector3 b0 = ref_up - kf.t * kf.t.dot(ref_up);
	const float blen = b0.length();
	if (blen < 1e-6f) return kf;
	kf.b0 = b0 / blen;
	kf.n0 = kf.t.cross(kf.b0);
	kf.tilt = static_cast<float>(c->get_point_tilt(ki));
	const float ct = std::cos(kf.tilt);
	const float st = std::sin(kf.tilt);
	kf.b =  kf.b0 * ct + kf.n0 * st;
	kf.n = -kf.b0 * st + kf.n0 * ct;
	kf.valid = true;
	return kf;
}

// Drag-handle radii — match the gizmo visualization so the handle sits
// at the visible ring's edge.
constexpr float TILT_HANDLE_RADIUS  = 0.10f;  // == frame_axis_len
constexpr float WIDTH_HANDLE_RADIUS = 0.08f;  // == width_ring_base_radius

// Walk the merged graph and emit `point_in` followed by `point_out`
// for every (curve, handle) pair. The order is deterministic so the
// secondary handle id maps back to the same TangentRef in every
// callback.
inline std::vector<TangentRef>
enumerate_tangent_refs(const curvenet::curvenet_builder::CurvenetGraph &g) {
	std::vector<TangentRef> out;
	for (std::size_t k = 0; k < g.knot_positions.size(); ++k) {
		for (const auto &ref : g.incidence[k]) {
			out.push_back({ ref.curve_id, ref.knot_idx, false });
			out.push_back({ ref.curve_id, ref.knot_idx, true  });
		}
	}
	return out;
}

// Collect the source mesh's vertex positions (pure C++ Vec3 list) so
// snap-to-surface code can do nearest-vertex lookup without re-touching
// godot-cpp on every call.
inline std::vector<curvenet::Vec3>
collect_source_mesh_positions(CurveNetDeformer3D *deformer) {
	std::vector<curvenet::Vec3> out;
	const NodePath src_path = deformer->get_source_path();
	Node *src_node = deformer->get_node_or_null(src_path);
	MeshInstance3D *src = Object::cast_to<MeshInstance3D>(src_node);
	if (src == nullptr) {
		return out;
	}
	Ref<Mesh> mesh = src->get_mesh();
	if (mesh.is_null()) {
		return out;
	}
	for (int s = 0; s < mesh->get_surface_count(); ++s) {
		Array arrays = mesh->surface_get_arrays(s);
		PackedVector3Array verts = arrays[Mesh::ARRAY_VERTEX];
		out.reserve(out.size() + static_cast<std::size_t>(verts.size()));
		for (int i = 0; i < verts.size(); ++i) {
			Vector3 v = verts[i];
			out.push_back({ v.x, v.y, v.z });
		}
	}
	return out;
}

inline Vector3 snap_to_nearest_vertex(const Vector3 &target,
                                        const std::vector<curvenet::Vec3> &mesh_positions) {
	if (mesh_positions.empty()) {
		return target;
	}
	const std::vector<curvenet::Vec3> probe = {{ target.x, target.y, target.z }};
	const auto pj = curvenet::surface_projection::project_to_vertices(probe, mesh_positions);
	if (pj.empty() || pj[0].mesh_index < 0) {
		return target;
	}
	return Vector3(static_cast<float>(pj[0].position.x),
	                static_cast<float>(pj[0].position.y),
	                static_cast<float>(pj[0].position.z));
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
                                               int32_t p_id, bool p_secondary) const {
	if (!p_secondary) {
		return String("knot ") + String::num_int64(p_id);
	}
	if (p_gizmo.is_valid()) {
		CurveNetDeformer3D *deformer =
			Object::cast_to<CurveNetDeformer3D>(p_gizmo->get_node_3d());
		if (deformer != nullptr) {
			const auto graph = build_graph_from_deformer(deformer);
			const TypedArray<Curve3D> curves = deformer->get_profile_curves();
			const std::size_t nT = enumerate_tangent_refs(graph).size();
			const std::size_t nK = enumerate_all_knots(curves).size();
			if (p_id >= 0 && static_cast<std::size_t>(p_id) < nT) {
				return String("tangent ") + String::num_int64(p_id);
			}
			if (p_id >= static_cast<int>(nT)
			    && static_cast<std::size_t>(p_id) < nT + nK) {
				return String("tilt ") + String::num_int64(p_id - nT);
			}
			if (p_id >= static_cast<int>(nT + nK)
			    && static_cast<std::size_t>(p_id) < nT + 2 * nK) {
				return String("width ") + String::num_int64(p_id - nT - nK);
			}
		}
	}
	return String("handle ") + String::num_int64(p_id);
}

Variant CurveNetGizmoPlugin::_get_handle_value(const Ref<EditorNode3DGizmo> &p_gizmo,
                                                  int32_t p_id, bool p_secondary) const {
	if (p_gizmo.is_null()) {
		return Variant();
	}
	CurveNetDeformer3D *deformer =
		Object::cast_to<CurveNetDeformer3D>(p_gizmo->get_node_3d());
	if (deformer == nullptr) {
		return Variant();
	}
	const auto graph = build_graph_from_deformer(deformer);
	const TypedArray<Curve3D> curves = deformer->get_profile_curves();

	if (!p_secondary) {
		if (p_id < 0 || static_cast<std::size_t>(p_id) >= graph.knot_positions.size()) {
			return Variant();
		}
		const curvenet::Vec3 &p = graph.knot_positions[p_id];
		return Vector3(static_cast<float>(p.x),
		                static_cast<float>(p.y),
		                static_cast<float>(p.z));
	}

	// Secondary handle id ranges:
	//   [0, nT)             — tangent in/out (Vector3 restore)
	//   [nT, nT+nK)         — tilt (float restore — radians)
	//   [nT+nK, nT+2*nK)    — width (float restore)
	const std::vector<TangentRef> tangents = enumerate_tangent_refs(graph);
	const std::size_t nT = tangents.size();
	const std::vector<KnotPosRef> all_knots = enumerate_all_knots(curves);
	const std::size_t nK = all_knots.size();
	if (p_id >= 0 && static_cast<std::size_t>(p_id) < nT) {
		const TangentRef &tref = tangents[p_id];
		Ref<Curve3D> c = curves[tref.curve_id];
		if (c.is_null()) return Variant();
		const Vector3 anchor = c->get_point_position(tref.knot_idx);
		const Vector3 offset = tref.is_out
			? c->get_point_out(tref.knot_idx)
			: c->get_point_in(tref.knot_idx);
		return anchor + offset;
	}
	if (p_id >= static_cast<int>(nT) && static_cast<std::size_t>(p_id) < nT + nK) {
		const KnotPosRef &kr = all_knots[p_id - nT];
		Ref<Curve3D> c = curves[kr.curve_id];
		if (c.is_null()) return Variant();
		return static_cast<float>(c->get_point_tilt(kr.knot_idx));
	}
	if (p_id >= static_cast<int>(nT + nK)
	    && static_cast<std::size_t>(p_id) < nT + 2 * nK) {
		const KnotPosRef &kr = all_knots[p_id - nT - nK];
		return static_cast<float>(deformer->get_knot_width(kr.curve_id, kr.knot_idx));
	}
	return Variant();
}

void CurveNetGizmoPlugin::_set_handle(const Ref<EditorNode3DGizmo> &p_gizmo,
                                        int32_t p_id, bool p_secondary,
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
	const TypedArray<Curve3D> curves = deformer->get_profile_curves();

	const Vector3 ray_from = p_camera->project_ray_origin(p_screen_pos);
	const Vector3 ray_dir  = p_camera->project_ray_normal(p_screen_pos);
	const Vector3 plane_normal = p_camera->get_camera_transform().basis.get_column(2);

	// Helper: ray vs camera-facing plane through `anchor_world`.
	auto raycast_to_plane = [&](const Vector3 &anchor_world) -> Vector3 {
		Plane plane(plane_normal, anchor_world);
		Array planes;
		planes.append(plane);
		PackedVector3Array hits = Geometry3D::get_singleton()->segment_intersects_convex(
			ray_from, ray_from + ray_dir * 16384.0f, planes);
		if (hits.is_empty()) {
			return anchor_world;
		}
		return hits[0];
	};

	if (!p_secondary) {
		// Primary handle: drag a merged knot. All incident Curve3D handles
		// follow, then snap the result to the closest source-mesh vertex.
		if (p_id < 0 || static_cast<std::size_t>(p_id) >= graph.knot_positions.size()) {
			return;
		}
		const curvenet::Vec3 &k = graph.knot_positions[p_id];
		const Vector3 anchor_world(static_cast<float>(k.x),
		                            static_cast<float>(k.y),
		                            static_cast<float>(k.z));
		Vector3 hit = raycast_to_plane(anchor_world);
		const std::vector<curvenet::Vec3> mesh_positions =
			collect_source_mesh_positions(deformer);
		hit = snap_to_nearest_vertex(hit, mesh_positions);
		for (const curvenet::curvenet_builder::KnotRef &ref : graph.incidence[p_id]) {
			Ref<Curve3D> c = curves[ref.curve_id];
			if (c.is_valid() && ref.knot_idx >= 0 && ref.knot_idx < c->get_point_count()) {
				c->set_point_position(ref.knot_idx, hit);
			}
		}
	} else {
		const std::vector<TangentRef> tangents = enumerate_tangent_refs(graph);
		const std::size_t nT = tangents.size();
		const std::vector<KnotPosRef> all_knots = enumerate_all_knots(curves);
		const std::size_t nK = all_knots.size();
		if (p_id >= 0 && static_cast<std::size_t>(p_id) < nT) {
			// Tangent drag (existing): free-float, no surface snap.
			const TangentRef &tref = tangents[p_id];
			Ref<Curve3D> c = curves[tref.curve_id];
			if (c.is_null()) return;
			const Vector3 anchor   = c->get_point_position(tref.knot_idx);
			const Vector3 cur_off  = tref.is_out ? c->get_point_out(tref.knot_idx)
			                                       : c->get_point_in(tref.knot_idx);
			const Vector3 anchor_world = anchor + cur_off;
			const Vector3 hit          = raycast_to_plane(anchor_world);
			const Vector3 new_off      = hit - anchor;
			if (tref.is_out) c->set_point_out(tref.knot_idx, new_off);
			else             c->set_point_in (tref.knot_idx, new_off);
		} else if (p_id >= static_cast<int>(nT)
		           && static_cast<std::size_t>(p_id) < nT + nK) {
			// Tilt drag: project hit into plane perpendicular to tangent at
			// origin, compute new angle = atan2(proj·n0, proj·b0).
			const KnotPosRef &kr = all_knots[p_id - nT];
			Ref<Curve3D> c = curves[kr.curve_id];
			const KnotFrame kf = compute_knot_frame(c, kr.knot_idx);
			if (!kf.valid) return;
			// Drag plane: perpendicular to tangent through origin (so the
			// raycast snaps to a point in the (b0, n0) plane).
			Plane drag_plane(kf.t, kf.origin);
			Array drag_planes; drag_planes.append(drag_plane);
			PackedVector3Array hits = Geometry3D::get_singleton()->segment_intersects_convex(
				ray_from, ray_from + ray_dir * 16384.0f, drag_planes);
			Vector3 hit = hits.is_empty() ? kf.origin : hits[0];
			Vector3 d = hit - kf.origin;
			const float proj_b = d.dot(kf.b0);
			const float proj_n = d.dot(kf.n0);
			if (proj_b * proj_b + proj_n * proj_n < 1e-12f) return;
			const float new_tilt = std::atan2(proj_n, proj_b);
			c->set_point_tilt(kr.knot_idx, new_tilt);
		} else if (p_id >= static_cast<int>(nT + nK)
		           && static_cast<std::size_t>(p_id) < nT + 2 * nK) {
			// Width drag: project hit into plane perpendicular to tangent at
			// origin, distance from origin / WIDTH_HANDLE_RADIUS = new w.
			const KnotPosRef &kr = all_knots[p_id - nT - nK];
			Ref<Curve3D> c = curves[kr.curve_id];
			const KnotFrame kf = compute_knot_frame(c, kr.knot_idx);
			if (!kf.valid) return;
			Plane drag_plane(kf.t, kf.origin);
			Array drag_planes; drag_planes.append(drag_plane);
			PackedVector3Array hits = Geometry3D::get_singleton()->segment_intersects_convex(
				ray_from, ray_from + ray_dir * 16384.0f, drag_planes);
			Vector3 hit = hits.is_empty() ? kf.origin : hits[0];
			Vector3 d = hit - kf.origin;
			const float radial = d.length();
			float new_w = radial / WIDTH_HANDLE_RADIUS;
			if (new_w < 0.05f) new_w = 0.05f;
			if (new_w > 50.0f) new_w = 50.0f;
			deformer->set_knot_width(kr.curve_id, kr.knot_idx, new_w);
		}
	}

	deformer->apply_deformation();
}

void CurveNetGizmoPlugin::_commit_handle(const Ref<EditorNode3DGizmo> &p_gizmo,
                                           int32_t p_id, bool p_secondary,
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

	if (p_cancel) {
		// User cancelled: roll the affected curve handles back to the
		// `p_restore` value the editor handed us at click time.
		if (!p_secondary) {
			if (p_id < 0 ||
					static_cast<std::size_t>(p_id) >= graph.knot_positions.size()) {
				return;
			}
			const Vector3 old = p_restore;
			for (const curvenet::curvenet_builder::KnotRef &ref : graph.incidence[p_id]) {
				Ref<Curve3D> c = curves[ref.curve_id];
				if (c.is_valid() && ref.knot_idx >= 0 && ref.knot_idx < c->get_point_count()) {
					c->set_point_position(ref.knot_idx, old);
				}
			}
		} else {
			const std::vector<TangentRef> tangents = enumerate_tangent_refs(graph);
			const std::size_t nT = tangents.size();
			const std::vector<KnotPosRef> all_knots = enumerate_all_knots(curves);
			const std::size_t nK = all_knots.size();
			if (p_id >= 0 && static_cast<std::size_t>(p_id) < nT) {
				const TangentRef &tref = tangents[p_id];
				Ref<Curve3D> c = curves[tref.curve_id];
				if (c.is_valid()) {
					const Vector3 old_world  = p_restore;
					const Vector3 anchor     = c->get_point_position(tref.knot_idx);
					const Vector3 old_offset = old_world - anchor;
					if (tref.is_out) c->set_point_out(tref.knot_idx, old_offset);
					else             c->set_point_in (tref.knot_idx, old_offset);
				}
			} else if (p_id >= static_cast<int>(nT)
			           && static_cast<std::size_t>(p_id) < nT + nK) {
				const KnotPosRef &kr = all_knots[p_id - nT];
				Ref<Curve3D> c = curves[kr.curve_id];
				if (c.is_valid()) {
					c->set_point_tilt(kr.knot_idx, static_cast<float>(p_restore));
				}
			} else if (p_id >= static_cast<int>(nT + nK)
			           && static_cast<std::size_t>(p_id) < nT + 2 * nK) {
				const KnotPosRef &kr = all_knots[p_id - nT - nK];
				deformer->set_knot_width(kr.curve_id, kr.knot_idx,
					static_cast<double>(static_cast<float>(p_restore)));
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

	if (!p_secondary) {
		if (p_id < 0 ||
				static_cast<std::size_t>(p_id) >= graph.knot_positions.size()) {
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
		return;
	}

	// Secondary handles: tangent / tilt / width — dispatch by id range.
	const std::vector<TangentRef> tangents = enumerate_tangent_refs(graph);
	const std::size_t nT = tangents.size();
	const std::vector<KnotPosRef> all_knots = enumerate_all_knots(curves);
	const std::size_t nK = all_knots.size();
	if (p_id >= 0 && static_cast<std::size_t>(p_id) < nT) {
		const TangentRef &tref = tangents[p_id];
		Ref<Curve3D> c = curves[tref.curve_id];
		if (c.is_null()) return;
		const Vector3 anchor      = c->get_point_position(tref.knot_idx);
		const Vector3 cur_off     = tref.is_out ? c->get_point_out(tref.knot_idx)
		                                          : c->get_point_in(tref.knot_idx);
		const Vector3 old_world   = p_restore;
		const Vector3 old_offset  = old_world - anchor;
		const String setter       = tref.is_out ? "set_point_out" : "set_point_in";
		undo->create_action("Move curvenet tangent " + String::num_int64(p_id));
		undo->add_do_method(c.ptr(),   setter, tref.knot_idx, cur_off);
		undo->add_undo_method(c.ptr(), setter, tref.knot_idx, old_offset);
		undo->add_do_method(deformer,   "apply_deformation");
		undo->add_undo_method(deformer, "apply_deformation");
		undo->commit_action(false);
	} else if (p_id >= static_cast<int>(nT)
	           && static_cast<std::size_t>(p_id) < nT + nK) {
		const KnotPosRef &kr = all_knots[p_id - nT];
		Ref<Curve3D> c = curves[kr.curve_id];
		if (c.is_null()) return;
		const float cur_tilt = c->get_point_tilt(kr.knot_idx);
		const float old_tilt = static_cast<float>(p_restore);
		undo->create_action("Tilt curvenet knot " + String::num_int64(p_id));
		undo->add_do_method(c.ptr(),   "set_point_tilt", kr.knot_idx, cur_tilt);
		undo->add_undo_method(c.ptr(), "set_point_tilt", kr.knot_idx, old_tilt);
		undo->add_do_method(deformer,   "apply_deformation");
		undo->add_undo_method(deformer, "apply_deformation");
		undo->commit_action(false);
	} else if (p_id >= static_cast<int>(nT + nK)
	           && static_cast<std::size_t>(p_id) < nT + 2 * nK) {
		const KnotPosRef &kr = all_knots[p_id - nT - nK];
		const double cur_w = deformer->get_knot_width(kr.curve_id, kr.knot_idx);
		const double old_w = static_cast<double>(static_cast<float>(p_restore));
		undo->create_action("Resize curvenet knot " + String::num_int64(p_id));
		undo->add_do_method(deformer,   "set_knot_width", kr.curve_id, kr.knot_idx, cur_w);
		undo->add_undo_method(deformer, "set_knot_width", kr.curve_id, kr.knot_idx, old_w);
		undo->add_do_method(deformer,   "apply_deformation");
		undo->add_undo_method(deformer, "apply_deformation");
		undo->commit_action(false);
	}
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
			handle_ids, false, /*secondary=*/false);
	}

	// Tangent handles (secondary): one each for point_in / point_out of
	// every (curve, knot_idx) reference in the graph. Index order matches
	// `enumerate_tangent_refs` so id ↔ TangentRef agrees across callbacks.
	const std::vector<TangentRef> tangents_drag = enumerate_tangent_refs(graph);
	PackedVector3Array tangent_handle_positions;
	PackedInt32Array   tangent_handle_ids;
	PackedVector3Array tangent_link_lines;
	tangent_handle_positions.resize(tangents_drag.size());
	tangent_handle_ids.resize(tangents_drag.size());
	for (std::size_t i = 0; i < tangents_drag.size(); ++i) {
		const TangentRef &tr = tangents_drag[i];
		Ref<Curve3D> tc = curves[tr.curve_id];
		if (tc.is_null()) {
			continue;
		}
		const Vector3 anchor = tc->get_point_position(tr.knot_idx);
		const Vector3 offset = tr.is_out
			? tc->get_point_out(tr.knot_idx)
			: tc->get_point_in(tr.knot_idx);
		const Vector3 world  = anchor + offset;
		tangent_handle_positions.set(i, world);
		tangent_handle_ids.set(i, static_cast<int>(i));
		tangent_link_lines.push_back(anchor);
		tangent_link_lines.push_back(world);
	}
	if (tangent_link_lines.size() > 0) {
		p_gizmo->add_lines(
			tangent_link_lines, get_material("tangent_link", p_gizmo), false,
			Color(1.0f, 0.85f, 0.3f, 0.6f));
	}
	if (tangent_handle_positions.size() > 0) {
		p_gizmo->add_handles(
			tangent_handle_positions, get_material("tangent_handles", p_gizmo),
			tangent_handle_ids, false, /*secondary=*/true);
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

	// DeGoes22 §3 per-knot frame visualization: short colored axis
	// segments at each knot showing the local (tangent, normal, binormal)
	// frame, with the normal/binormal rotated by the artist's tilt
	// (Curve3D::get_point_tilt) around the tangent. Makes tilt visible.
	// Plus an octagonal width ring perpendicular to tangent, scaled by
	// the per-knot width from `knot_widths` (default 1.0).
	const float frame_axis_len = 0.10f;
	const float width_ring_base_radius = 0.08f;
	PackedVector3Array frame_t_lines;
	PackedVector3Array frame_n_lines;
	PackedVector3Array frame_b_lines;
	PackedVector3Array width_ring_lines;
	const TypedArray<PackedFloat32Array> widths_arr = deformer->get_knot_widths();
	for (int ci = 0; ci < curves.size(); ++ci) {
		Ref<Curve3D> c = curves[ci];
		if (c.is_null()) continue;
		const int n = c->get_point_count();
		PackedFloat32Array curve_widths;
		if (ci < widths_arr.size()) curve_widths = widths_arr[ci];
		for (int ki = 0; ki < n; ++ki) {
			const Vector3 origin = c->get_point_position(ki);
			// Tangent: prefer outgoing, fall back to incoming.
			Vector3 t_dir;
			if (ki + 1 < n) {
				t_dir = c->get_point_position(ki + 1) - origin;
			} else if (ki - 1 >= 0) {
				t_dir = origin - c->get_point_position(ki - 1);
			} else {
				t_dir = Vector3(1.0f, 0.0f, 0.0f);
			}
			const float tlen = t_dir.length();
			if (tlen < 1e-6f) continue;
			const Vector3 t = t_dir / tlen;
			// Pick a reference up vector to derive an initial binormal —
			// world Y when |t·Y| < 0.95, else world Z. Gram-Schmidt to
			// get a plane perpendicular to t.
			Vector3 ref_up(0.0f, 1.0f, 0.0f);
			if (std::fabs(t.dot(ref_up)) > 0.95f) {
				ref_up = Vector3(0.0f, 0.0f, 1.0f);
			}
			Vector3 b0 = ref_up - t * t.dot(ref_up);
			const float blen = b0.length();
			if (blen < 1e-6f) continue;
			b0 = b0 / blen;
			Vector3 n0 = t.cross(b0);
			// Apply the artist's tilt: rotate (n0, b0) around t by tilt.
			const float tilt = static_cast<float>(c->get_point_tilt(ki));
			const float ct = std::cos(tilt);
			const float st = std::sin(tilt);
			const Vector3 b = b0 * ct + n0 * st;
			const Vector3 nv = -b0 * st + n0 * ct;
			frame_t_lines.push_back(origin);
			frame_t_lines.push_back(origin + t  * frame_axis_len);
			frame_n_lines.push_back(origin);
			frame_n_lines.push_back(origin + nv * frame_axis_len);
			frame_b_lines.push_back(origin);
			frame_b_lines.push_back(origin + b  * frame_axis_len);
			// Width ring: octagon in the (b, n) plane scaled by width w.
			float w = 1.0f;
			if (ki < curve_widths.size()) w = curve_widths[ki];
			const float r = width_ring_base_radius * w;
			constexpr int RING_SEG = 8;
			Vector3 prev = origin + b * r;
			for (int s = 1; s <= RING_SEG; ++s) {
				const float ang = static_cast<float>(s) * (6.28318530718f / RING_SEG);
				const Vector3 cur = origin + (b * std::cos(ang) + nv * std::sin(ang)) * r;
				width_ring_lines.push_back(prev);
				width_ring_lines.push_back(cur);
				prev = cur;
			}
		}
	}
	if (frame_t_lines.size() > 0) {
		p_gizmo->add_lines(frame_t_lines, get_material("frame_t", p_gizmo), false,
			Color(1.0f, 0.3f, 0.3f, 1.0f));
	}
	if (frame_n_lines.size() > 0) {
		p_gizmo->add_lines(frame_n_lines, get_material("frame_n", p_gizmo), false,
			Color(0.3f, 1.0f, 0.3f, 1.0f));
	}
	if (frame_b_lines.size() > 0) {
		p_gizmo->add_lines(frame_b_lines, get_material("frame_b", p_gizmo), false,
			Color(0.3f, 0.5f, 1.0f, 1.0f));
	}
	if (width_ring_lines.size() > 0) {
		p_gizmo->add_lines(width_ring_lines, get_material("width_ring", p_gizmo), false,
			Color(0.3f, 1.0f, 0.7f, 0.6f));
	}

	// Drag handles for tilt + width — placed in the secondary handle space
	// after the tangent handles. Id encoding:
	//   [0, nT)             — tangent in/out (existing)
	//   [nT, nT+nK)         — tilt handle per (curve, knot)
	//   [nT+nK, nT+2*nK)    — width handle per (curve, knot)
	// where nT = tangents_drag.size() and nK = enumerate_all_knots(...).size().
	const std::vector<KnotPosRef> all_knots = enumerate_all_knots(curves);
	const std::size_t nT = tangents_drag.size();
	const std::size_t nK = all_knots.size();
	PackedVector3Array tilt_handle_positions;
	PackedInt32Array   tilt_handle_ids;
	PackedVector3Array width_handle_positions;
	PackedInt32Array   width_handle_ids;
	for (std::size_t i = 0; i < nK; ++i) {
		const KnotPosRef &kr = all_knots[i];
		Ref<Curve3D> c = curves[kr.curve_id];
		const KnotFrame kf = compute_knot_frame(c, kr.knot_idx);
		if (!kf.valid) continue;
		// Tilt handle: at the rotated b axis, distance TILT_HANDLE_RADIUS.
		// Drag → rotate around tangent.
		const Vector3 tilt_world = kf.origin + kf.b * TILT_HANDLE_RADIUS;
		tilt_handle_positions.push_back(tilt_world);
		tilt_handle_ids.push_back(static_cast<int>(nT + i));
		// Width handle: at the rotated b axis, distance WIDTH_HANDLE_RADIUS · w.
		// Drag → resize. Always offset along the same direction so it doesn't
		// collide with the tilt handle.
		float w_cur = 1.0f;
		if (kr.curve_id < widths_arr.size()) {
			PackedFloat32Array row = widths_arr[kr.curve_id];
			if (kr.knot_idx < row.size()) w_cur = row[kr.knot_idx];
		}
		const Vector3 width_world = kf.origin - kf.b * WIDTH_HANDLE_RADIUS * w_cur;
		width_handle_positions.push_back(width_world);
		width_handle_ids.push_back(static_cast<int>(nT + nK + i));
	}
	if (tilt_handle_positions.size() > 0) {
		p_gizmo->add_handles(
			tilt_handle_positions, get_material("tilt_handles", p_gizmo),
			tilt_handle_ids, false, /*secondary=*/true);
	}
	if (width_handle_positions.size() > 0) {
		p_gizmo->add_handles(
			width_handle_positions, get_material("width_handles", p_gizmo),
			width_handle_ids, false, /*secondary=*/true);
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
