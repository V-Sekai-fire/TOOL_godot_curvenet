// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
#include "vertex_handles_editor_plugin.h"

#include "vertex_handles_gizmo_plugin.h"

#include <godot_cpp/core/class_db.hpp>

namespace godot {

void VertexHandlesEditorPlugin::_bind_methods() {
}

void VertexHandlesEditorPlugin::_enter_tree() {
	gizmo_plugin.instantiate();
	gizmo_plugin->set_editor_plugin(this);
	add_node_3d_gizmo_plugin(gizmo_plugin);
}

void VertexHandlesEditorPlugin::_exit_tree() {
	if (gizmo_plugin.is_valid()) {
		remove_node_3d_gizmo_plugin(gizmo_plugin);
		gizmo_plugin.unref();
	}
}

} // namespace godot
