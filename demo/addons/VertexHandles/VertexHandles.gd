@tool
class_name VertexHandles extends Node3D

## Add a new [VertexHandles] as a child of [MeshInstance3D] to modify it's mesh



# PUBLIC

@export var wireframe := true : 
	set(value):
		wireframe = value
		_request_redraw.emit()
@export var wireframe_color := Color.CORNFLOWER_BLUE : 
	set(value):
		wireframe_color = value
		_request_redraw.emit()

# Array of point arrays
@export var point_arrays := [] : set = _set_points

# The target mesh whose vertices we want to modify
@onready var mesh = get_parent().mesh


# PRIVATE

signal _request_redraw

# TODO
# - Add options
# - Support double vertices
# - Support multiple surfaces
# - Fix handle rendering bug
# - Sort handles from back to front

func _ready() -> void:
	assert(get_parent() is MeshInstance3D)

	_refresh_point_arrays()

	# Refresh the points just in case we are dealing with a new mesh
	EditorInterface.get_selection().selection_changed.connect(_refresh_point_arrays)
	# Wire to a CurveNetDeformer3D sibling so handle moves trigger redeform.
	# Look up by class so we don't hardcode a node name; falls back silently
	# if the user hasn't placed one in the scene yet.
	var deformer := _find_curvenet_deformer()
	if deformer != null:
		_request_redraw.connect(deformer.apply_deformation)

func _find_curvenet_deformer() -> Node:
	# Walk siblings + parent's siblings looking for a CurveNetDeformer3D.
	var p := get_parent()
	if p == null:
		return null
	for child in p.get_children():
		if child.get_class() == "CurveNetDeformer3D":
			return child
	var gp := p.get_parent()
	if gp == null:
		return null
	for child in gp.get_children():
		if child.get_class() == "CurveNetDeformer3D":
			return child
	return null

func _refresh_point_arrays():
	if not self in EditorInterface.get_selection().get_selected_nodes():
		return
	
	get_parent().mesh = _to_array_mesh(get_parent().mesh)
	mesh = get_parent().mesh
	
	point_arrays = []
	
	for i in range(0,mesh.get_surface_count()):
		var arrays = mesh.surface_get_arrays(i)
		
		point_arrays.push_back(arrays[Mesh.ARRAY_VERTEX])

func _update_mesh():
	if is_node_ready() and not point_arrays.is_empty():
		mesh = get_parent().mesh
		
		var surface_arrays := []
		for i in range(0,mesh.get_surface_count()):
			var arrays = mesh.surface_get_arrays(i)
			surface_arrays.push_back(arrays)
		
		# Cache the primitive type
		var type : Mesh.PrimitiveType = _get_primitive_type(mesh)
		
		mesh.clear_surfaces()
		
		for i in range(0,surface_arrays.size()):
			surface_arrays[i][Mesh.ARRAY_VERTEX] = PackedVector3Array( point_arrays[i] )
			
			mesh.add_surface_from_arrays(type, surface_arrays[i])
	
	_request_redraw.emit()

func _set_points(value):
	point_arrays = value
	_update_mesh()

func set_point(i:int, point_idx:int, p:Vector3):
	point_arrays[i][point_idx] = p
	_update_mesh()

func _to_array_mesh(_mesh:Mesh) -> ArrayMesh:
	var surface_arrays := []
	for i in range(0,_mesh.get_surface_count()):
		var arrays = _mesh.surface_get_arrays(i)
		surface_arrays.push_back(arrays)
	
	# Cache the primitive type
	var type : Mesh.PrimitiveType = _get_primitive_type(_mesh)
	
	_mesh = ArrayMesh.new()
	
	for i in range(0,surface_arrays.size()):
		_mesh.add_surface_from_arrays(type, surface_arrays[i])
	
	return _mesh

func _get_primitive_type(_mesh:Mesh, id:=0) -> Mesh.PrimitiveType:
	return RenderingServer.mesh_get_surface(_mesh.get_rid(), 0)["primitive"]
