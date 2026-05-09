// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
#include "curvenet_editor_plugin.h"

#include "curvenet_gizmo_plugin.h"

#include <godot_cpp/core/class_db.hpp>

namespace godot {

void CurveNetEditorPlugin::_bind_methods() {
}

void CurveNetEditorPlugin::_enter_tree() {
	gizmo_plugin.instantiate();
	gizmo_plugin->set_editor_plugin(this);
	add_node_3d_gizmo_plugin(gizmo_plugin);
}

void CurveNetEditorPlugin::_exit_tree() {
	if (gizmo_plugin.is_valid()) {
		remove_node_3d_gizmo_plugin(gizmo_plugin);
		gizmo_plugin.unref();
	}
}

} // namespace godot
