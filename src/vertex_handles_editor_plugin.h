// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
#ifndef VERTEX_HANDLES_EDITOR_PLUGIN_H
#define VERTEX_HANDLES_EDITOR_PLUGIN_H

#include <godot_cpp/classes/editor_plugin.hpp>

namespace godot {

class VertexHandlesGizmoPlugin;

// Native port of demo/addons/VertexHandles/plugin.gd. Registers
// VertexHandlesGizmoPlugin with the 3D editor on enter_tree and tears it
// down on exit_tree.
class VertexHandlesEditorPlugin : public EditorPlugin {
	GDCLASS(VertexHandlesEditorPlugin, EditorPlugin)

	Ref<VertexHandlesGizmoPlugin> gizmo_plugin;

protected:
	static void _bind_methods();

public:
	VertexHandlesEditorPlugin() = default;
	~VertexHandlesEditorPlugin() = default;

	void _enter_tree() override;
	void _exit_tree() override;
};

} // namespace godot

#endif
