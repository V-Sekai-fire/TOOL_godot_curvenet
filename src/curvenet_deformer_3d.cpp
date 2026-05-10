// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
#include "curvenet_deformer_3d.h"

#include "curvenet/curvenet_builder.h"
#include "curvenet/cut_mesh.h"
#include "curvenet/cut_mesh_laplacian.h"
#include "curvenet/deform_solve.h"
#include "curvenet/incomplete_cholesky.h"
#include "curvenet/halfedge.h"
#include "curvenet/halfedge_builder.h"
#include "curvenet/harmonic_solve.h"
#include "curvenet/scaled_frames.h"
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

	ClassDB::bind_method(D_METHOD("set_knot_widths", "widths"), &CurveNetDeformer3D::set_knot_widths);
	ClassDB::bind_method(D_METHOD("get_knot_widths"), &CurveNetDeformer3D::get_knot_widths);
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "knot_widths", PROPERTY_HINT_ARRAY_TYPE,
							  String::num_int64(Variant::PACKED_FLOAT32_ARRAY)),
			"set_knot_widths", "get_knot_widths");
	ClassDB::bind_method(D_METHOD("set_knot_width", "curve_id", "knot_idx", "w"),
		&CurveNetDeformer3D::set_knot_width);
	ClassDB::bind_method(D_METHOD("get_knot_width", "curve_id", "knot_idx"),
		&CurveNetDeformer3D::get_knot_width);

	ClassDB::bind_method(D_METHOD("set_length_tiebreak", "v"), &CurveNetDeformer3D::set_length_tiebreak);
	ClassDB::bind_method(D_METHOD("get_length_tiebreak"), &CurveNetDeformer3D::get_length_tiebreak);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "length_tiebreak", PROPERTY_HINT_RANGE, "0.0,1.0,0.01"),
			"set_length_tiebreak", "get_length_tiebreak");

	ClassDB::bind_method(D_METHOD("set_deformation_active", "v"), &CurveNetDeformer3D::set_deformation_active);
	ClassDB::bind_method(D_METHOD("is_deformation_active"), &CurveNetDeformer3D::is_deformation_active);
	ClassDB::bind_method(D_METHOD("set_use_incomplete_cholesky", "v"), &CurveNetDeformer3D::set_use_incomplete_cholesky);
	ClassDB::bind_method(D_METHOD("get_use_incomplete_cholesky"), &CurveNetDeformer3D::get_use_incomplete_cholesky);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "deformation_active"),
			"set_deformation_active", "is_deformation_active");

	ClassDB::bind_method(D_METHOD("set_use_direct_delta_mush", "v"), &CurveNetDeformer3D::set_use_direct_delta_mush);
	ClassDB::bind_method(D_METHOD("get_use_direct_delta_mush"), &CurveNetDeformer3D::get_use_direct_delta_mush);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "use_direct_delta_mush"),
			"set_use_direct_delta_mush", "get_use_direct_delta_mush");
	ClassDB::bind_method(D_METHOD("set_ddm_top_k", "v"), &CurveNetDeformer3D::set_ddm_top_k);
	ClassDB::bind_method(D_METHOD("get_ddm_top_k"), &CurveNetDeformer3D::get_ddm_top_k);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "ddm_top_k", PROPERTY_HINT_RANGE, "1,32,1"),
			"set_ddm_top_k", "get_ddm_top_k");
	ClassDB::bind_method(D_METHOD("set_ddm_smooth_iters", "v"), &CurveNetDeformer3D::set_ddm_smooth_iters);
	ClassDB::bind_method(D_METHOD("get_ddm_smooth_iters"), &CurveNetDeformer3D::get_ddm_smooth_iters);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "ddm_smooth_iters", PROPERTY_HINT_RANGE, "0,32,1"),
			"set_ddm_smooth_iters", "get_ddm_smooth_iters");

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
	rest_cache.prev_Fv.clear();
	rest_cache.prev_Xv.clear();
	rest_cache.prev_solve_valid = false;
	rest_cache.nc = 0;
	rest_cache.source_hash = 0;
	rest_cache.ddm_influences.clear();
	rest_cache.ddm_built = false;
	rest_cache.rest_curve_knots.clear();
	rest_cache.rest_curve_tilts.clear();
	rest_cache.rest_curve_widths.clear();
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

void CurveNetDeformer3D::set_knot_widths(const TypedArray<PackedFloat32Array> &p_widths) {
	knot_widths = p_widths;
	// Widths feed into per-handle F_h at runtime; rest widths are baked
	// at the next rebuild. Don't invalidate on mere edit — the artist may
	// be tweaking widths interactively.
}

TypedArray<PackedFloat32Array> CurveNetDeformer3D::get_knot_widths() const {
	return knot_widths;
}

void CurveNetDeformer3D::set_knot_width(int p_curve_id, int p_knot_idx, double p_w) {
	if (p_curve_id < 0 || p_knot_idx < 0) return;
	while (knot_widths.size() <= p_curve_id) {
		knot_widths.push_back(PackedFloat32Array{});
	}
	PackedFloat32Array row = knot_widths[p_curve_id];
	while (row.size() <= p_knot_idx) {
		row.push_back(1.0f);
	}
	row.set(p_knot_idx, static_cast<float>(p_w));
	knot_widths[p_curve_id] = row;
}

double CurveNetDeformer3D::get_knot_width(int p_curve_id, int p_knot_idx) const {
	if (p_curve_id < 0 || p_curve_id >= knot_widths.size()) return 1.0;
	PackedFloat32Array row = knot_widths[p_curve_id];
	if (p_knot_idx < 0 || p_knot_idx >= row.size()) return 1.0;
	return static_cast<double>(row[p_knot_idx]);
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

void CurveNetDeformer3D::set_use_incomplete_cholesky(bool p_v) {
	use_incomplete_cholesky = p_v;
}

bool CurveNetDeformer3D::get_use_incomplete_cholesky() const {
	return use_incomplete_cholesky;
}

void CurveNetDeformer3D::set_use_direct_delta_mush(bool p_v) {
	if (use_direct_delta_mush != p_v) {
		// Toggle changes which bind-time work the cache holds; force a
		// rebuild on next `apply_deformation` so we don't carry stale
		// influences (or lack thereof) into the new mode.
		rest_cache.ddm_influences.clear();
		rest_cache.ddm_built = false;
	}
	use_direct_delta_mush = p_v;
}

bool CurveNetDeformer3D::get_use_direct_delta_mush() const {
	return use_direct_delta_mush;
}

void CurveNetDeformer3D::set_ddm_top_k(int p_v) {
	if (p_v < 1) p_v = 1;
	if (ddm_top_k != p_v) {
		rest_cache.ddm_influences.clear();
		rest_cache.ddm_built = false;
	}
	ddm_top_k = p_v;
}

int CurveNetDeformer3D::get_ddm_top_k() const {
	return ddm_top_k;
}

void CurveNetDeformer3D::set_ddm_smooth_iters(int p_v) {
	if (p_v < 0) p_v = 0;
	if (ddm_smooth_iters != p_v) {
		rest_cache.ddm_influences.clear();
		rest_cache.ddm_built = false;
	}
	ddm_smooth_iters = p_v;
}

int CurveNetDeformer3D::get_ddm_smooth_iters() const {
	return ddm_smooth_iters;
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

		// Sparse rest-pose matrices via Sharp & Crane 2020 intrinsic
		// mollification — finite cot weights even on character meshes
		// with degenerate triangles. The embedding-based path produces
		// ±∞ on real input (Mire body) and CG fails to converge.
		const double mollify_delta =
			curvenet::cut_mesh_laplacian::default_mollify_delta(
				rest_cache.positions, rest_cache.tri_indices);
		rest_cache.Lh_csr =
			curvenet::cut_mesh_laplacian::assemble_lh_csr_robust(
				rest_cache.cut_mesh, rest_cache.positions, mollify_delta);
		rest_cache.LhsM_csr =
			curvenet::cut_mesh_laplacian::assemble_vt_lh_v_csr_robust(
				rest_cache.cut_mesh, rest_cache.positions, mollify_delta);

		// Bake rest curve knot positions + tilts for the runtime deformation
		// gradient computation. The artist's CURRENT Curve3D state at
		// bind time is the rest pose; subsequent drags treat it as posed.
		rest_cache.rest_curve_knots.clear();
		rest_cache.rest_curve_tilts.clear();
		rest_cache.rest_curve_widths.clear();
		rest_cache.rest_curve_knots.reserve(profile_curves.size());
		rest_cache.rest_curve_tilts.reserve(profile_curves.size());
		rest_cache.rest_curve_widths.reserve(profile_curves.size());
		for (int i = 0; i < profile_curves.size(); ++i) {
			Ref<Curve3D> c = profile_curves[i];
			std::vector<curvenet::Vec3> rk;
			std::vector<double>         rt;
			std::vector<double>         rw;
			if (c.is_valid()) {
				const int n = c->get_point_count();
				rk.reserve(n);
				rt.reserve(n);
				rw.reserve(n);
				PackedFloat32Array curve_widths;
				if (i < knot_widths.size()) {
					curve_widths = knot_widths[i];
				}
				for (int j = 0; j < n; ++j) {
					Vector3 p = c->get_point_position(j);
					rk.push_back({ p.x, p.y, p.z });
					rt.push_back(static_cast<double>(c->get_point_tilt(j)));
					double wj = 1.0;
					if (j < curve_widths.size()) {
						wj = static_cast<double>(curve_widths[j]);
					}
					rw.push_back(wj);
				}
			}
			rest_cache.rest_curve_knots.push_back(std::move(rk));
			rest_cache.rest_curve_tilts.push_back(std::move(rt));
			rest_cache.rest_curve_widths.push_back(std::move(rw));
		}

		rest_cache.source_hash = hash_val;
		rest_cache.valid = true;
	}

	const std::size_t nv = rest_cache.positions.size();
	const std::size_t nh = rest_cache.cut_mesh.he_count();
	const int nc = rest_cache.nc;

	// Bind-time DDM harvest: scalar harmonic solve per handle gives the
	// per-vertex weight matrix W. Smooth (Laplacian iters) + sparsify
	// (top-K) prepares the runtime LBS-style matvec. Runs once per
	// topology change OR DDM toggle flip — `set_use_direct_delta_mush`
	// invalidates `ddm_built` so we re-harvest on next apply.
	if (use_direct_delta_mush && rest_cache.valid && !rest_cache.ddm_built && nc > 0) {
		const std::size_t ncs = static_cast<std::size_t>(nc);
		// Identity Fc: each handle column is one-hot in its own row.
		std::vector<double> Fc_id(ncs * ncs, 0.0);
		for (std::size_t i = 0; i < ncs; ++i) {
			Fc_id[i * ncs + i] = 1.0;
		}
		const auto sample_col_pack = [](int curve_id, int /*sample_idx*/, bool /*side*/) -> int {
			return curve_id;
		};
		// solve_multi → nv × nc row-major; column h is handle h's harmonic response.
		const std::vector<double> W_dense =
			curvenet::harmonic_solve::solve_multi(
				rest_cache.cut_mesh, rest_cache.positions, sample_col_pack, Fc_id, ncs);
		curvenet::direct_delta_mush::WeightMatrix W(nv,
			std::vector<double>(ncs, 0.0));
		for (std::size_t v = 0; v < nv; ++v) {
			for (std::size_t h = 0; h < ncs; ++h) {
				W[v][h] = W_dense[v * ncs + h];
			}
		}
		// Sample-promoted vertices have zero-row from the §6 solve (their
		// V column is empty); override to one-hot so they follow their
		// handle exactly at runtime.
		for (std::size_t v = 0; v < nv; ++v) {
			const auto &k = rest_cache.cut_mesh.vertex_kind[v];
			if (k.tag == curvenet::cut_mesh::CutVertexKindTag::sample) {
				const int col = k.curve_id;
				for (auto &x : W[v]) x = 0.0;
				if (col >= 0 && static_cast<std::size_t>(col) < ncs) {
					W[v][col] = 1.0;
				}
			}
		}
		// Vertex-vertex adjacency from the triangle index list.
		curvenet::direct_delta_mush::Adjacency adj(nv);
		auto add_edge = [&](int a, int b) {
			if (a < 0 || b < 0) return;
			for (int x : adj[a]) if (x == b) return;
			adj[a].push_back(b);
		};
		for (std::size_t f = 0; f + 2 < rest_cache.tri_indices.size(); f += 3) {
			const int a = rest_cache.tri_indices[f + 0];
			const int b = rest_cache.tri_indices[f + 1];
			const int c = rest_cache.tri_indices[f + 2];
			add_edge(a, b); add_edge(b, a);
			add_edge(b, c); add_edge(c, b);
			add_edge(c, a); add_edge(a, c);
		}
		const auto Wsmoothed = curvenet::direct_delta_mush::smooth_weights(
			W, adj, static_cast<std::size_t>(ddm_smooth_iters), 0.5);
		rest_cache.ddm_influences = curvenet::direct_delta_mush::sparsify_top_k(
			Wsmoothed, static_cast<std::size_t>(ddm_top_k));
		rest_cache.ddm_built = true;
	}

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

	// DDM runtime branch: skip the §6 solve, do a sparse LBS matvec per vertex
	// using the bind-time-harvested weights. Per-handle 4x4 transforms are
	// built from DeGoes22 §3 isolated-segment deformation gradients — captures
	// rotation + length scale, not just translation.
	if (use_direct_delta_mush && rest_cache.ddm_built) {
		std::vector<curvenet::direct_delta_mush::Mat4> transforms;
		transforms.reserve(static_cast<std::size_t>(nc));
		for (int c = 0; c < nc; ++c) {
			const auto &rest_p = rest_cache.col_rest_pos[c];
			Vector3 posed_p_v(static_cast<float>(Xc[c * 3 + 0]),
			                    static_cast<float>(Xc[c * 3 + 1]),
			                    static_cast<float>(Xc[c * 3 + 2]));
			const curvenet::Vec3 posed_p{ posed_p_v.x, posed_p_v.y, posed_p_v.z };

			// Default to identity rotation if we can't bracket a segment
			// (single-point curves or out-of-bounds knot indices).
			curvenet::scaled_frames::Mat3 F = curvenet::scaled_frames::mat3_identity();
			const auto &handle = rest_cache.col_input_handle[c];
			const int curve_id = handle.first;
			const int knot_idx = handle.second;
			if (curve_id >= 0 && curve_id < static_cast<int>(rest_cache.rest_curve_knots.size())
			    && knot_idx >= 0
			    && curve_id < profile_curves.size()) {
				const auto &rest_knots = rest_cache.rest_curve_knots[curve_id];
				Ref<Curve3D> curve = profile_curves[curve_id];
				if (curve.is_valid() && static_cast<std::size_t>(knot_idx) < rest_knots.size()
				    && knot_idx < curve->get_point_count()) {
					// Pick a "directional partner" knot — outgoing if not last,
					// else incoming.
					int partner_idx = -1;
					if (knot_idx + 1 < static_cast<int>(rest_knots.size())
					    && knot_idx + 1 < curve->get_point_count()) {
						partner_idx = knot_idx + 1;
					} else if (knot_idx - 1 >= 0) {
						partner_idx = knot_idx - 1;
					}
					if (partner_idx >= 0) {
						const curvenet::Vec3 rest_q = rest_knots[partner_idx];
						Vector3 pq_v = curve->get_point_position(partner_idx);
						const curvenet::Vec3 posed_q{ pq_v.x, pq_v.y, pq_v.z };
						F = curvenet::scaled_frames::isolated_segment_gradient(
							rest_p, rest_q, posed_p, posed_q);
					}

					// DeGoes22 §3 width: scale perpendicular to the posed
					// tangent by s_w = w_posed / w_rest, identity along the
					// tangent. Implements the body-cross-section knob.
					const auto &rest_widths = rest_cache.rest_curve_widths[curve_id];
					double s_w = 1.0;
					if (static_cast<std::size_t>(knot_idx) < rest_widths.size()
					    && rest_widths[knot_idx] > 1e-9) {
						double w_posed = 1.0;
						if (curve_id < knot_widths.size()) {
							PackedFloat32Array curve_widths = knot_widths[curve_id];
							if (knot_idx < curve_widths.size()) {
								w_posed = static_cast<double>(curve_widths[knot_idx]);
							}
						}
						s_w = w_posed / rest_widths[knot_idx];
					}
					if (std::fabs(s_w - 1.0) > 1e-9) {
						curvenet::Vec3 t_axis{ 1.0, 0.0, 0.0 };
						if (knot_idx + 1 < curve->get_point_count()) {
							Vector3 q = curve->get_point_position(knot_idx + 1);
							t_axis = { q.x - posed_p.x, q.y - posed_p.y, q.z - posed_p.z };
						} else if (knot_idx - 1 >= 0) {
							Vector3 q = curve->get_point_position(knot_idx - 1);
							t_axis = { posed_p.x - q.x, posed_p.y - q.y, posed_p.z - q.z };
						}
						const double tn = std::sqrt(t_axis.x * t_axis.x + t_axis.y * t_axis.y + t_axis.z * t_axis.z);
						if (tn > 1e-12) {
							t_axis = { t_axis.x / tn, t_axis.y / tn, t_axis.z / tn };
							// F_perp = s_w · I + (1 - s_w) · t⊗tᵀ
							const double k = 1.0 - s_w;
							curvenet::scaled_frames::Mat3 F_perp =
								curvenet::scaled_frames::mat3_make(
									s_w + k * t_axis.x * t_axis.x, k * t_axis.x * t_axis.y, k * t_axis.x * t_axis.z,
									k * t_axis.y * t_axis.x, s_w + k * t_axis.y * t_axis.y, k * t_axis.y * t_axis.z,
									k * t_axis.z * t_axis.x, k * t_axis.z * t_axis.y, s_w + k * t_axis.z * t_axis.z);
							F = curvenet::scaled_frames::mat3_mul(F_perp, F);
						}
					}

					// DeGoes22 §3 tilt: rotation around the posed tangent by
					// (posed_tilt - rest_tilt). Captures artist's twist of the
					// local frame at this knot. Composed BEFORE the tangent
					// rotation so the final F is R_tilt · F_segment.
					const auto &rest_tilts = rest_cache.rest_curve_tilts[curve_id];
					if (static_cast<std::size_t>(knot_idx) < rest_tilts.size()) {
						const double rest_tilt  = rest_tilts[knot_idx];
						const double posed_tilt = static_cast<double>(curve->get_point_tilt(knot_idx));
						const double dtilt = posed_tilt - rest_tilt;
						if (std::fabs(dtilt) > 1e-9) {
							const double dx = posed_p.x - rest_p.x;  // posed-tangent fallback
							curvenet::Vec3 axis{ dx, 0.0, 0.0 };
							// Use the actual posed tangent if we have a partner.
							if (knot_idx + 1 < curve->get_point_count()) {
								Vector3 q = curve->get_point_position(knot_idx + 1);
								axis = { q.x - posed_p.x, q.y - posed_p.y, q.z - posed_p.z };
							} else if (knot_idx - 1 >= 0) {
								Vector3 q = curve->get_point_position(knot_idx - 1);
								axis = { posed_p.x - q.x, posed_p.y - q.y, posed_p.z - q.z };
							}
							const double an = std::sqrt(axis.x * axis.x + axis.y * axis.y + axis.z * axis.z);
							if (an > 1e-12) {
								axis = { axis.x / an, axis.y / an, axis.z / an };
								const double s = std::sin(dtilt);
								const double cc = std::cos(dtilt);
								const curvenet::scaled_frames::Mat3 K =
									curvenet::scaled_frames::skew(axis);
								const curvenet::scaled_frames::Mat3 K2 =
									curvenet::scaled_frames::mat3_mul(K, K);
								// Rodrigues: R = I + sin(θ)·K + (1 - cos(θ))·K²
								curvenet::scaled_frames::Mat3 R_tilt =
									curvenet::scaled_frames::mat3_add(
										curvenet::scaled_frames::mat3_identity(),
										curvenet::scaled_frames::mat3_add(
											curvenet::scaled_frames::mat3_smul(s, K),
											curvenet::scaled_frames::mat3_smul(1.0 - cc, K2)));
								F = curvenet::scaled_frames::mat3_mul(R_tilt, F);
							}
						}
					}
				}
			}

			// T_h(x) = F · (x - rest_p) + posed_p
			//        = F · x + (posed_p - F · rest_p)
			const curvenet::Vec3 F_rest =
				curvenet::scaled_frames::mat3_mul_vec(F, rest_p);
			const double tx = posed_p.x - F_rest.x;
			const double ty = posed_p.y - F_rest.y;
			const double tz = posed_p.z - F_rest.z;
			curvenet::direct_delta_mush::Mat4 T = {
				F[0], F[1], F[2], tx,
				F[3], F[4], F[5], ty,
				F[6], F[7], F[8], tz,
				0.0,  0.0,  0.0,  1.0
			};
			transforms.push_back(T);
		}
		PackedVector3Array deformed_vertices;
		deformed_vertices.resize(static_cast<int>(nv));
		for (std::size_t v = 0; v < nv; ++v) {
			const auto &rest_p = rest_cache.positions[v];
			const auto &infl = rest_cache.ddm_influences[v];
			const auto p = curvenet::direct_delta_mush::lbs_matvec(
				transforms, infl, rest_p.x, rest_p.y, rest_p.z);
			deformed_vertices.set(static_cast<int>(v),
				Vector3(static_cast<float>(p[0]),
				         static_cast<float>(p[1]),
				         static_cast<float>(p[2])));
		}
		emit_mesh(deformed_vertices);
		return;
	}

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

	// Lazily build the ICC(0) factor of LhsM_csr the first frame the
	// flag is observed true. Factorisation is amortised across all
	// subsequent solves so the per-frame path is just two backsolves
	// per CG iter (same cost as one A·x mat-vec).
	if (use_incomplete_cholesky && !rest_cache.incomplete_cholesky_built) {
		rest_cache.incomplete_cholesky_factor = curvenet::incomplete_cholesky::factor_with_retry(
			rest_cache.LhsM_csr, &rest_cache.incomplete_cholesky_shift);
		rest_cache.incomplete_cholesky_built = !rest_cache.incomplete_cholesky_factor.breakdown;
	}

	// Multi-RHS sparse CG (one column at a time, sharing the cached
	// CSR matrix + preconditioner across columns). Warm-starts each
	// column from the previous frame's iterate when available.
	// Switches between D-Jacobi PCG (default) and ICC(0)-PCG (loop
	// 100/3, gated on `use_incomplete_cholesky` and successful factorisation).
	const std::size_t cg_max_iter = std::max<std::size_t>(50, nv * 2);
	const bool icc_active = use_incomplete_cholesky && rest_cache.incomplete_cholesky_built;
	auto cg_multi = [&](std::size_t k, const std::vector<double> &rhs,
	                     const std::vector<double> *prev) {
		std::vector<double> X(nv * k, 0.0);
		std::vector<double> b_col(nv, 0.0);
		std::vector<double> x0_col(nv, 0.0);
		const bool have_prev = (prev != nullptr) && (prev->size() == nv * k);
		for (std::size_t c = 0; c < k; ++c) {
			for (std::size_t i = 0; i < nv; ++i) {
				b_col[i] = rhs[i * k + c];
				x0_col[i] = have_prev ? (*prev)[i * k + c] : 0.0;
			}
			const std::vector<double> x_col = icc_active
				? curvenet::incomplete_cholesky::cg_icc_with_guess(
				      rest_cache.LhsM_csr, rest_cache.incomplete_cholesky_factor,
				      b_col, x0_col, cg_max_iter, 1e-8)
				: curvenet::sparse::cg_with_guess(
				      rest_cache.LhsM_csr, b_col, x0_col,
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
	const std::vector<double> Fv = cg_multi(9, rhs_a,
		rest_cache.prev_solve_valid ? &rest_cache.prev_Fv : nullptr);

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
	const std::vector<double> Xv = cg_multi(3, rhs_b,
		rest_cache.prev_solve_valid ? &rest_cache.prev_Xv : nullptr);

	rest_cache.prev_Fv = Fv;
	rest_cache.prev_Xv = Xv;
	rest_cache.prev_solve_valid = true;

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
