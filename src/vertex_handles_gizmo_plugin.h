// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
#ifndef VERTEX_HANDLES_GIZMO_PLUGIN_H
#define VERTEX_HANDLES_GIZMO_PLUGIN_H

#include <godot_cpp/classes/camera3d.hpp>
#include <godot_cpp/classes/editor_node3d_gizmo.hpp>
#include <godot_cpp/classes/editor_node3d_gizmo_plugin.hpp>
#include <godot_cpp/classes/editor_plugin.hpp>
#include <godot_cpp/classes/node3d.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/variant.hpp>
#include <godot_cpp/variant/vector2.hpp>

namespace godot {

// Native port of demo/addons/VertexHandles/VertexHandlesGizmo.gd. Renders a
// wireframe of the parent mesh and a draggable handle per vertex; clicks +
// drags route through `VertexHandles3D::set_point`. Lives only in the editor.
class VertexHandlesGizmoPlugin : public EditorNode3DGizmoPlugin {
	GDCLASS(VertexHandlesGizmoPlugin, EditorNode3DGizmoPlugin)

	EditorPlugin *editor_plugin = nullptr;

protected:
	static void _bind_methods();

public:
	VertexHandlesGizmoPlugin();
	~VertexHandlesGizmoPlugin() = default;

	void set_editor_plugin(EditorPlugin *p_plugin);

	bool _has_gizmo(Node3D *p_node) const override;
	Ref<EditorNode3DGizmo> _create_gizmo(Node3D *p_node) const override;
	String _get_gizmo_name() const override;
	String _get_handle_name(const Ref<EditorNode3DGizmo> &p_gizmo, int32_t p_id, bool p_secondary) const override;
	Variant _get_handle_value(const Ref<EditorNode3DGizmo> &p_gizmo, int32_t p_id, bool p_secondary) const override;
	void _set_handle(const Ref<EditorNode3DGizmo> &p_gizmo, int32_t p_id, bool p_secondary, Camera3D *p_camera, const Vector2 &p_screen_pos) override;
	void _commit_handle(const Ref<EditorNode3DGizmo> &p_gizmo, int32_t p_id, bool p_secondary, const Variant &p_restore, bool p_cancel) override;
	void _redraw(const Ref<EditorNode3DGizmo> &p_gizmo) override;
};

} // namespace godot

#endif
