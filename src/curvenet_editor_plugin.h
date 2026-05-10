// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
#ifndef CURVENET_EDITOR_PLUGIN_H
#define CURVENET_EDITOR_PLUGIN_H

#include <godot_cpp/classes/editor_plugin.hpp>

namespace godot {

class CurveNetGizmoPlugin;

// Registers the CurveNetGizmoPlugin with the 3D editor on enter_tree
// and tears it down on exit_tree, mirroring the
// VertexHandlesEditorPlugin pattern.
class CurveNetEditorPlugin : public EditorPlugin {
	GDCLASS(CurveNetEditorPlugin, EditorPlugin)

	Ref<CurveNetGizmoPlugin> gizmo_plugin;

protected:
	static void _bind_methods();

public:
	CurveNetEditorPlugin() = default;
	~CurveNetEditorPlugin() = default;

	void _enter_tree() override;
	void _exit_tree() override;
};

} // namespace godot

#endif
