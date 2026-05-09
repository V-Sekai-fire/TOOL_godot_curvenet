// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
#ifndef CURVENET_GIZMO_PLUGIN_H
#define CURVENET_GIZMO_PLUGIN_H

#include <godot_cpp/classes/editor_node3d_gizmo.hpp>
#include <godot_cpp/classes/editor_node3d_gizmo_plugin.hpp>
#include <godot_cpp/classes/editor_plugin.hpp>
#include <godot_cpp/classes/node3d.hpp>
#include <godot_cpp/variant/string.hpp>

namespace godot {

// Editor gizmo for CurveNetDeformer3D nodes. Renders the merged
// curvenet graph (built via curvenet_builder + profile_curves) so the
// authoring user can see:
//   * spheres at every merged knot, coloured by KnotKind (anchor =
//     red, regular = white, intersection = green)
//   * tangent rays at intersections (yellow) — the inputs to the
//     §3 corner-normal computation
//   * cyan link-lines from each merged knot to the mesh vertex it
//     projects onto (the surface_projection result)
//
// The gizmo redraws when its node emits `_curvenet_redraw_request`
// (CurveNetDeformer3D fires this at the end of apply_deformation).
class CurveNetGizmoPlugin : public EditorNode3DGizmoPlugin {
	GDCLASS(CurveNetGizmoPlugin, EditorNode3DGizmoPlugin)

	EditorPlugin *editor_plugin = nullptr;

protected:
	static void _bind_methods();

public:
	CurveNetGizmoPlugin();
	~CurveNetGizmoPlugin() = default;

	void set_editor_plugin(EditorPlugin *p_plugin);

	bool                 _has_gizmo(Node3D *p_node) const override;
	Ref<EditorNode3DGizmo> _create_gizmo(Node3D *p_node) const override;
	String               _get_gizmo_name() const override;
	void                 _redraw(const Ref<EditorNode3DGizmo> &p_gizmo) override;
};

} // namespace godot

#endif
