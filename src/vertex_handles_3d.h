// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
#ifndef VERTEX_HANDLES_3D_H
#define VERTEX_HANDLES_3D_H

#include <godot_cpp/classes/array_mesh.hpp>
#include <godot_cpp/classes/mesh.hpp>
#include <godot_cpp/classes/mesh_instance3d.hpp>
#include <godot_cpp/variant/color.hpp>
#include <godot_cpp/variant/packed_vector3_array.hpp>
#include <godot_cpp/variant/typed_array.hpp>

namespace godot {

// Native port of demo/addons/VertexHandles/VertexHandles.gd. Extends
// MeshInstance3D directly so it owns the mesh being edited; the gizmo plugin
// renders draggable handles for each vertex of `mesh`.
class VertexHandles3D : public MeshInstance3D {
	GDCLASS(VertexHandles3D, MeshInstance3D)

	bool wireframe = true;
	Color wireframe_color = Color(0.39f, 0.58f, 0.93f, 1.0f); // CORNFLOWER_BLUE
	TypedArray<PackedVector3Array> point_arrays;

protected:
	static void _bind_methods();

public:
	VertexHandles3D() = default;
	~VertexHandles3D() = default;

	void set_wireframe(bool p_v);
	bool get_wireframe() const;

	void set_wireframe_color(const Color &p_c);
	Color get_wireframe_color() const;

	void set_point_arrays(const TypedArray<PackedVector3Array> &p_arrays);
	TypedArray<PackedVector3Array> get_point_arrays() const;

	void _ready() override;

	// Move a single handle and rebuild this node's mesh.
	void set_point(int p_surface, int p_point_idx, const Vector3 &p_p);

	// Rebuild point_arrays from the current `mesh` surfaces.
	void refresh_point_arrays();

	// Rebuild `mesh` from the current point_arrays.
	void update_mesh();

private:
	Node *find_curvenet_deformer() const;
	Ref<ArrayMesh> to_array_mesh(const Ref<Mesh> &p_mesh) const;
	Mesh::PrimitiveType primitive_type_of(const Ref<Mesh> &p_mesh) const;
};

} // namespace godot

#endif
