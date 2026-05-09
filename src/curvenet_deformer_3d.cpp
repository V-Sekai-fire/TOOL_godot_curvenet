// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
#include "curvenet_deformer_3d.h"

#include "curvenet/curvenet_builder.h"
#include "curvenet/cut_mesh.h"
#include "curvenet/cut_mesh_laplacian.h"
#include "curvenet/deform_solve.h"
#include "curvenet/halfedge.h"
#include "curvenet/halfedge_builder.h"
#include "curvenet/harmonic_solve.h"
#include "curvenet/sparse_linalg.h"
#include "curvenet/surface_projection.h"
#include "curvenet/vec3.h"

#include <godot_cpp/classes/array_mesh.hpp>
#include <godot_cpp/classes/curve3d.hpp>
#include <godot_cpp/classes/mesh.hpp>
#include <godot_cpp/classes/mesh_instance3d.hpp>
#include <godot_cpp/classes/surface_tool.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/packed_int32_array.hpp>
#include <godot_cpp/variant/packed_vector3_array.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

namespace godot {

void CurveNetDeformer3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_source_path", "path"), &CurveNetDeformer3D::set_source_path);
	ClassDB::bind_method(D_METHOD("get_source_path"), &CurveNetDeformer3D::get_source_path);
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "source_path"), "set_source_path", "get_source_path");

	ClassDB::bind_method(D_METHOD("set_profile_curves", "curves"), &CurveNetDeformer3D::set_profile_curves);
	ClassDB::bind_method(D_METHOD("get_profile_curves"), &CurveNetDeformer3D::get_profile_curves);
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "profile_curves", PROPERTY_HINT_ARRAY_TYPE,
							  String::num_int64(Variant::OBJECT) + "/" + String::num_int64(PROPERTY_HINT_RESOURCE_TYPE) + ":Curve3D"),
			"set_profile_curves", "get_profile_curves");

	ClassDB::bind_method(D_METHOD("set_length_tiebreak", "v"), &CurveNetDeformer3D::set_length_tiebreak);
	ClassDB::bind_method(D_METHOD("get_length_tiebreak"), &CurveNetDeformer3D::get_length_tiebreak);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "length_tiebreak", PROPERTY_HINT_RANGE, "0.0,1.0,0.01"),
			"set_length_tiebreak", "get_length_tiebreak");

	ClassDB::bind_method(D_METHOD("set_deformation_active", "v"), &CurveNetDeformer3D::set_deformation_active);
	ClassDB::bind_method(D_METHOD("is_deformation_active"), &CurveNetDeformer3D::is_deformation_active);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "deformation_active"),
			"set_deformation_active", "is_deformation_active");

	ClassDB::bind_method(D_METHOD("apply_deformation"), &CurveNetDeformer3D::apply_deformation);
	ClassDB::bind_method(D_METHOD("get_face_count"), &CurveNetDeformer3D::get_face_count);
	ClassDB::bind_method(D_METHOD("get_face_vertex_count", "face_index"), &CurveNetDeformer3D::get_face_vertex_count);
	ClassDB::bind_method(D_METHOD("evaluate_face", "face_index", "s", "t"), &CurveNetDeformer3D::evaluate_face);

	ADD_SIGNAL(MethodInfo("_curvenet_redraw_request"));
}

void CurveNetDeformer3D::invalidate_cache() {
	rest_cache.valid = false;
	rest_cache.positions.clear();
	rest_cache.tri_indices.clear();
	rest_cache.cut_mesh = curvenet::cut_mesh::CutMesh{};
	rest_cache.col_rest_pos.clear();
	rest_cache.col_input_handle.clear();
	rest_cache.Lh_csr   = curvenet::sparse::SparseMatrixCSR{};
	rest_cache.LhsM_csr = curvenet::sparse::SparseMatrixCSR{};
	rest_cache.nc = 0;
	rest_cache.source_hash = 0;
}

void CurveNetDeformer3D::set_source_path(const NodePath &p_path) {
	source_path = p_path;
	invalidate_cache();
}

NodePath CurveNetDeformer3D::get_source_path() const {
	return source_path;
}

void CurveNetDeformer3D::set_profile_curves(const TypedArray<Curve3D> &p_curves) {
	profile_curves = p_curves;
	// Profile curves don't affect the rest-pose cache (only bindings do); no
	// invalidation needed.
}

TypedArray<Curve3D> CurveNetDeformer3D::get_profile_curves() const {
	return profile_curves;
}

void CurveNetDeformer3D::set_length_tiebreak(double p_v) {
	length_tiebreak = p_v;
	invalidate_cache(); // tri-to-quad weights depend on length_tiebreak
}

double CurveNetDeformer3D::get_length_tiebreak() const {
	return length_tiebreak;
}

void CurveNetDeformer3D::set_deformation_active(bool p_v) {
	deformation_active = p_v;
	// Properties are set during scene load before the node is in the tree, so
	// `get_node_or_null(source_path)` would resolve to null and emit a spurious
	// error. Defer the first apply until `_ready` once the scene is wired up.
	if (deformation_active && is_inside_tree()) {
		apply_deformation();
	}
}

bool CurveNetDeformer3D::is_deformation_active() const {
	return deformation_active;
}

void CurveNetDeformer3D::_ready() {
	if (deformation_active) {
		apply_deformation();
	}
}

void CurveNetDeformer3D::apply_deformation() {
	Node *src_node = get_node_or_null(source_path);
	MeshInstance3D *src = Object::cast_to<MeshInstance3D>(src_node);
	if (src == nullptr) {
		UtilityFunctions::printerr("CurveNetDeformer3D: source_path does not point to a MeshInstance3D");
		return;
	}
	Ref<Mesh> mesh = src->get_mesh();
	if (mesh.is_null()) {
		return;
	}

	// Hash source-mesh vertex+index data + profile_curves count so we can
	// detect authoring edits and rebuild the rest-pose cache lazily.
	std::uint64_t hash_val = static_cast<std::uint64_t>(mesh->get_rid().get_id());
	hash_val = hash_val * 1099511628211ULL + static_cast<std::uint64_t>(mesh->get_surface_count());
	for (int s = 0; s < mesh->get_surface_count(); ++s) {
		Array arrays = mesh->surface_get_arrays(s);
		PackedVector3Array verts = arrays[Mesh::ARRAY_VERTEX];
		PackedInt32Array indices = arrays[Mesh::ARRAY_INDEX];
		hash_val = hash_val * 1099511628211ULL + static_cast<std::uint64_t>(verts.size());
		hash_val = hash_val * 1099511628211ULL + static_cast<std::uint64_t>(indices.size());
	}
	hash_val = hash_val * 1099511628211ULL + static_cast<std::uint64_t>(profile_curves.size());

	if (!rest_cache.valid || rest_cache.source_hash != hash_val) {
		// Rebuild rest-pose: extract triangles + positions, build halfedge
		// mesh, build curvenet from profile_curves, project knots to mesh
		// vertices, promote them to samples in a fresh CutMesh.
		rest_cache.positions.clear();
		rest_cache.tri_indices.clear();
		for (int s = 0; s < mesh->get_surface_count(); ++s) {
			Array arrays = mesh->surface_get_arrays(s);
			PackedVector3Array verts = arrays[Mesh::ARRAY_VERTEX];
			PackedInt32Array indices = arrays[Mesh::ARRAY_INDEX];
			const int v_offset = static_cast<int>(rest_cache.positions.size());
			for (int i = 0; i < verts.size(); ++i) {
				Vector3 v = verts[i];
				rest_cache.positions.push_back({ v.x, v.y, v.z });
			}
			if (indices.size() > 0) {
				for (int i = 0; i + 2 < indices.size(); i += 3) {
					rest_cache.tri_indices.push_back(v_offset + indices[i]);
					rest_cache.tri_indices.push_back(v_offset + indices[i + 1]);
					rest_cache.tri_indices.push_back(v_offset + indices[i + 2]);
				}
			} else {
				for (int i = 0; i + 2 < verts.size(); i += 3) {
					rest_cache.tri_indices.push_back(v_offset + i);
					rest_cache.tri_indices.push_back(v_offset + i + 1);
					rest_cache.tri_indices.push_back(v_offset + i + 2);
				}
			}
		}

		const std::size_t nv = rest_cache.positions.size();
		curvenet::HalfedgeMesh hm =
			curvenet::halfedge_builder::from_triangles(nv, rest_cache.tri_indices);

		// Fresh CutMesh: every vertex starts as mesh_vertex.
		rest_cache.cut_mesh = curvenet::cut_mesh::CutMesh{};
		rest_cache.cut_mesh.base = hm;
		rest_cache.cut_mesh.vertex_kind.assign(
			nv, curvenet::cut_mesh::CutVertexKind::mesh_vertex_kind());
		rest_cache.cut_mesh.segment_of_halfedge.assign(hm.he_count(), -1);

		// Build curvenet from profile_curves' rest knot positions.
		std::vector<curvenet::curvenet_builder::CurvePoints> input_curves;
		input_curves.reserve(profile_curves.size());
		for (int i = 0; i < profile_curves.size(); ++i) {
			Ref<Curve3D> c = profile_curves[i];
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
		const curvenet::curvenet_builder::CurvenetGraph cg =
			curvenet::curvenet_builder::build(input_curves, 1.0e-6);

		// Project each merged curvenet knot to its closest mesh vertex.
		const auto projections = curvenet::surface_projection::project_to_vertices(
			cg.knot_positions, rest_cache.positions);
		const std::vector<int> knot_to_col =
			curvenet::surface_projection::promote_vertex_samples(
				rest_cache.cut_mesh, projections);

		// Build per-column rest position + (curve_id, handle_idx) lookup.
		// The first knot mapped to each column wins.
		const int max_col = [&]() {
			int m = -1;
			for (int c : knot_to_col) {
				if (c > m) {
					m = c;
				}
			}
			return m;
		}();
		rest_cache.nc = max_col + 1;
		rest_cache.col_rest_pos.assign(rest_cache.nc, curvenet::Vec3{ 0.0, 0.0, 0.0 });
		rest_cache.col_input_handle.assign(rest_cache.nc, std::make_pair(-1, -1));
		std::vector<bool> col_set(rest_cache.nc, false);
		for (std::size_t k = 0; k < knot_to_col.size(); ++k) {
			const int col = knot_to_col[k];
			if (col < 0 || col_set[col]) {
				continue;
			}
			const auto &incidence = cg.incidence[k];
			if (incidence.empty()) {
				continue;
			}
			const auto &ref = incidence.front();
			rest_cache.col_input_handle[col] =
				std::make_pair(ref.curve_id, ref.knot_idx);
			rest_cache.col_rest_pos[col] = projections[k].position;
			col_set[col] = true;
		}

		// Sparse rest-pose matrices: O(nh) memory + O(nh) bind work
		// instead of the dense O(nh²) explosion (see docs/PERF_BASELINE.md).
		rest_cache.Lh_csr =
			curvenet::cut_mesh_laplacian::assemble_lh_csr(
				rest_cache.cut_mesh, rest_cache.positions);
		rest_cache.LhsM_csr =
			curvenet::cut_mesh_laplacian::assemble_vt_lh_v_csr(
				rest_cache.cut_mesh, rest_cache.positions);

		rest_cache.source_hash = hash_val;
		rest_cache.valid = true;
	}

	const std::size_t nv = rest_cache.positions.size();
	const std::size_t nh = rest_cache.cut_mesh.he_count();
	const int nc = rest_cache.nc;

	// Helper that emits the deformed PackedVector3Array as an ArrayMesh
	// directly (bypasses SurfaceTool::generate_normals's vertex splits,
	// preserves the source mesh's vertex count). Normals are computed
	// by averaging incident triangle face normals per vertex.
	auto emit_mesh = [&](const PackedVector3Array &deformed_vertices) {
		PackedVector3Array normals;
		normals.resize(deformed_vertices.size());
		for (int i = 0; i < normals.size(); ++i) {
			normals.set(i, Vector3(0.0f, 0.0f, 0.0f));
		}
		PackedInt32Array indices;
		indices.resize(rest_cache.tri_indices.size());
		for (std::size_t i = 0; i < rest_cache.tri_indices.size(); ++i) {
			indices.set(static_cast<int>(i), rest_cache.tri_indices[i]);
		}
		// Accumulate face normals onto incident vertices.
		for (std::size_t f = 0; f + 2 < rest_cache.tri_indices.size(); f += 3) {
			const int a = rest_cache.tri_indices[f + 0];
			const int b = rest_cache.tri_indices[f + 1];
			const int c = rest_cache.tri_indices[f + 2];
			if (a < 0 || b < 0 || c < 0) {
				continue;
			}
			const Vector3 va = deformed_vertices[a];
			const Vector3 vb = deformed_vertices[b];
			const Vector3 vc = deformed_vertices[c];
			const Vector3 fn = (vb - va).cross(vc - va);
			normals.set(a, normals[a] + fn);
			normals.set(b, normals[b] + fn);
			normals.set(c, normals[c] + fn);
		}
		for (int i = 0; i < normals.size(); ++i) {
			Vector3 n = normals[i];
			const float len = n.length();
			normals.set(i, len > 0.0f ? n / len : Vector3(0.0f, 1.0f, 0.0f));
		}
		Array arrays;
		arrays.resize(Mesh::ARRAY_MAX);
		arrays[Mesh::ARRAY_VERTEX] = deformed_vertices;
		arrays[Mesh::ARRAY_NORMAL] = normals;
		arrays[Mesh::ARRAY_INDEX]  = indices;
		Ref<ArrayMesh> out_mesh;
		out_mesh.instantiate();
		out_mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, arrays);
		set_mesh(out_mesh);
	};

	if (nc == 0) {
		// No curvenet samples — emit the source mesh unchanged.
		PackedVector3Array verts;
		verts.resize(static_cast<int>(nv));
		for (std::size_t v = 0; v < nv; ++v) {
			const curvenet::Vec3 &p = rest_cache.positions[v];
			verts.set(static_cast<int>(v),
				Vector3(static_cast<float>(p.x), static_cast<float>(p.y), static_cast<float>(p.z)));
		}
		emit_mesh(verts);
		return;
	}

	// Per-frame: assemble Fc (identity per sample) and Xc (current handle
	// positions from profile_curves).
	std::vector<double> Fc(nc * 9, 0.0);
	std::vector<double> Xc(nc * 3, 0.0);
	for (int c = 0; c < nc; ++c) {
		Fc[c * 9 + 0] = 1.0;
		Fc[c * 9 + 4] = 1.0;
		Fc[c * 9 + 8] = 1.0;
		const auto &handle = rest_cache.col_input_handle[c];
		curvenet::Vec3 pos = rest_cache.col_rest_pos[c];
		if (handle.first >= 0 && handle.first < profile_curves.size()) {
			Ref<Curve3D> curve = profile_curves[handle.first];
			if (curve.is_valid() &&
					handle.second >= 0 &&
					handle.second < curve->get_point_count()) {
				Vector3 p = curve->get_point_position(handle.second);
				pos = { p.x, p.y, p.z };
			}
		}
		Xc[c * 3 + 0] = pos.x;
		Xc[c * 3 + 1] = pos.y;
		Xc[c * 3 + 2] = pos.z;
	}

	// `surface_projection::promote_vertex_samples` packs the column index
	// into vertex_kind.curve_id with sample_idx = 0, so the sample-column
	// packer just passes the curve_id through.
	const auto sample_col = [](int curve_id, int /*sample_idx*/, bool /*side*/) -> int {
		return curve_id;
	};

	// Helper that applies Vᵀ implicitly: scatter per-halfedge values
	// onto the target mesh-vertex (skip sample-promoted halfedges).
	auto apply_vt = [&](const std::vector<double> &Y_he, std::size_t k) {
		std::vector<double> out(nv * k, 0.0);
		for (std::size_t h = 0; h < nh; ++h) {
			const int col = curvenet::cut_mesh::v_column_of(rest_cache.cut_mesh, h);
			if (col < 0) {
				continue;
			}
			for (std::size_t c = 0; c < k; ++c) {
				out[static_cast<std::size_t>(col) * k + c] += Y_he[h * k + c];
			}
		}
		return out;
	};

	// Multi-RHS sparse CG (one column at a time, sharing the cached
	// CSR matrix + diagonal preconditioner across columns).
	const std::size_t cg_max_iter = std::max<std::size_t>(50, nv * 2);
	auto cg_multi = [&](std::size_t k, const std::vector<double> &rhs) {
		std::vector<double> X(nv * k, 0.0);
		std::vector<double> b_col(nv, 0.0);
		for (std::size_t c = 0; c < k; ++c) {
			for (std::size_t i = 0; i < nv; ++i) {
				b_col[i] = rhs[i * k + c];
			}
			const std::vector<double> x_col =
				curvenet::sparse::cg(rest_cache.LhsM_csr, b_col,
				                       cg_max_iter, 1e-8);
			for (std::size_t i = 0; i < nv; ++i) {
				X[i * k + c] = x_col[i];
			}
		}
		return X;
	};

	// Stage 1 (Eq. 6a): solve for Fv (per-vertex deformation gradients,
	// 9 columns). Reuses cached Lh_csr, LhsM_csr.
	const std::vector<double> CFc =
		curvenet::harmonic_solve::compute_c_fc_matrix(
			rest_cache.cut_mesh, sample_col, Fc, 9);
	const std::vector<double> Lh_CFc =
		curvenet::sparse::spmv_multi(rest_cache.Lh_csr, CFc, 9);
	const std::vector<double> Vt_Lh_CFc = apply_vt(Lh_CFc, 9);
	std::vector<double> rhs_a(nv * 9, 0.0);
	for (std::size_t i = 0; i < nv * 9; ++i) {
		rhs_a[i] = -Vt_Lh_CFc[i];
	}
	const std::vector<double> Fv = cg_multi(9, rhs_a);

	// Bridge (Eq. 3 + per-face average): build F_f and y_h.
	const std::vector<double> Fh =
		curvenet::deform_solve::compute_fh(
			rest_cache.cut_mesh, sample_col, Fv, Fc, 9);
	const std::vector<double> Ff =
		curvenet::deform_solve::average_over_faces(rest_cache.cut_mesh, Fh, 9);
	const std::vector<double> yh =
		curvenet::deform_solve::compute_yh(rest_cache.cut_mesh, rest_cache.positions, Ff);

	// Stage 2 (Eq. 6b): solve for Xv (vertex positions, 3 columns).
	const std::vector<double> CXc =
		curvenet::harmonic_solve::compute_c_fc_matrix(
			rest_cache.cut_mesh, sample_col, Xc, 3);
	std::vector<double> diff(nh * 3, 0.0);
	for (std::size_t i = 0; i < nh * 3; ++i) {
		diff[i] = CXc[i] - yh[i];
	}
	const std::vector<double> Lh_diff =
		curvenet::sparse::spmv_multi(rest_cache.Lh_csr, diff, 3);
	const std::vector<double> Vt_Lh_diff = apply_vt(Lh_diff, 3);
	std::vector<double> rhs_b(nv * 3, 0.0);
	for (std::size_t i = 0; i < nv * 3; ++i) {
		rhs_b[i] = -Vt_Lh_diff[i];
	}
	const std::vector<double> Xv = cg_multi(3, rhs_b);

	// Emit the deformed mesh — bypassing SurfaceTool's generate_normals
	// vertex splits keeps the original vertex count and preserves index
	// correspondence with the cut-mesh.
	PackedVector3Array verts;
	verts.resize(static_cast<int>(nv));
	for (std::size_t v = 0; v < nv; ++v) {
		const auto &kind = rest_cache.cut_mesh.vertex_kind[v];
		if (kind.tag == curvenet::cut_mesh::CutVertexKindTag::sample) {
			const int col = kind.curve_id;
			verts.set(static_cast<int>(v), Vector3(
				static_cast<float>(Xc[col * 3 + 0]),
				static_cast<float>(Xc[col * 3 + 1]),
				static_cast<float>(Xc[col * 3 + 2])));
		} else {
			verts.set(static_cast<int>(v), Vector3(
				static_cast<float>(Xv[v * 3 + 0]),
				static_cast<float>(Xv[v * 3 + 1]),
				static_cast<float>(Xv[v * 3 + 2])));
		}
	}
	emit_mesh(verts);

	UtilityFunctions::print(
		"CurveNetDeformer3D: ",
		static_cast<int>(profile_curves.size()), " profiles, ",
		nc, " sample columns, ",
		static_cast<int>(nv), " verts deformed");

	// Tell the editor gizmo (if present) to refresh.
	emit_signal("_curvenet_redraw_request");
}

int CurveNetDeformer3D::get_face_count() const {
	return rest_cache.valid
		? static_cast<int>(rest_cache.cut_mesh.base.face_count)
		: 0;
}

int CurveNetDeformer3D::get_face_vertex_count(int /*face_index*/) const {
	// New pipeline operates on triangle-only cut-meshes; every face has
	// exactly 3 halfedges in its loop.
	return rest_cache.valid ? 3 : 0;
}

Vector3 CurveNetDeformer3D::evaluate_face(int /*face_index*/, double /*s*/, double /*t*/) {
	// Per-face Coons evaluation belonged to the old tris-to-quads stub. The
	// DeGoes22 pipeline doesn't expose per-face evaluation — the deformation
	// is global through the §4.3 solve. This entry point is retained for
	// API compatibility; callers should sample the deformed mesh directly.
	return Vector3{ 0.0f, 0.0f, 0.0f };
}

} // namespace godot
