// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
#ifndef CURVENET_DEFORMER_3D_H
#define CURVENET_DEFORMER_3D_H

#include "curvenet/cut_mesh.h"
#include "curvenet/dense_linalg.h"
#include "curvenet/direct_delta_mush.h"
#include "curvenet/halfedge.h"
#include "curvenet/incomplete_cholesky.h"
#include "curvenet/sparse_linalg.h"
#include "curvenet/vec3.h"

#include <cstdint>
#include <utility>
#include <vector>

#include <godot_cpp/classes/curve3d.hpp>
#include <godot_cpp/classes/mesh_instance3d.hpp>
#include <godot_cpp/variant/node_path.hpp>
#include <godot_cpp/variant/typed_array.hpp>

namespace godot {

// CurveNetDeformer3D drives a source mesh's vertices via Pixar-style profile
// curves following DeGoes22 (`DeGoes2022Curvenet` in references.bib). The
// runtime ε-merges shared endpoints across all `profile_curves`, projects
// each merged knot to its closest mesh vertex (sample-promoting it), then
// runs §4.3's two-stage harmonic / Poisson solve to deform the mesh.
//
// Authoring: drop a `CurveNetDeformer3D` next to your mesh and assign one
// or more `Curve3D` resources to `profile_curves`. The associated
// `CurveNetGizmoPlugin` adds in-editor handles for the curvenet knots and
// the per-handle tangents so users can drag them; dragging snaps the knot
// to the nearest source-mesh vertex and re-runs the solver.
class CurveNetDeformer3D : public MeshInstance3D {
	GDCLASS(CurveNetDeformer3D, MeshInstance3D)

	NodePath source_path;
	TypedArray<Curve3D> profile_curves;
	double length_tiebreak = 0.1;
	bool deformation_active = false;
	// Incomplete-Cholesky preconditioner opt-in. Off by default;
	// diagonal-Jacobi remains the production code path. Used by
	// benchmarks (and eventually the runtime) once it's measured to
	// be a wall-clock win on the real workload, not just the
	// synthetic 81k diag.
	bool use_incomplete_cholesky = false;

	// Direct Delta Mush opt-in (Le & Lewis 2019, curvenet-adapted).
	// Off by default. When true, bind-time harvests per-handle harmonic
	// response weights once and the per-frame path becomes a sparse
	// LBS-style matvec (no §6 solves at runtime). Required for the
	// Quest 3 0.8 ms / 50k-vert budget; on desktop the §6 path with
	// warm-started PCG is competitive and remains the default.
	bool use_direct_delta_mush = false;
	int  ddm_top_k        = 4;
	int  ddm_smooth_iters = 3;

	// Cache of the rest-pose pipeline: halfedge mesh + sample-promoted
	// CutMesh, plus the column ↔ input-handle mapping needed at runtime
	// to pull per-frame sample positions from the user's Curve3D handles.
	//
	// The big perf win: every matrix in this struct depends only on the
	// rest pose. We assemble them ONCE at bind time and the per-frame
	// `apply_deformation` only computes RHS vectors + solves. That cuts
	// the dominant dense-matMul cost (`Lₕ V`, `Vᵀ Lₕ V`) out of the hot
	// loop entirely.
	struct RestCache {
		bool                                  valid       = false;
		std::vector<curvenet::Vec3>           positions;        // rest mesh vertex positions
		std::vector<int>                      tri_indices;      // flat index list, 3 ints/face
		curvenet::cut_mesh::CutMesh           cut_mesh;         // sample-promoted
		std::vector<curvenet::Vec3>           col_rest_pos;     // rest position per sample column
		std::vector<std::pair<int, int>>      col_input_handle; // (curve_id, handle_idx) per column
		// Pre-assembled rest-pose matrices, sparse (CSR). Replaces the
		// dense `Lh, V, Vt, LhS, LhsM` blob — at character mesh sizes
		// the dense path is O(nh²) memory and explodes past ~1k verts.
		curvenet::sparse::SparseMatrixCSR     Lh_csr;           // nh × nh
		curvenet::sparse::SparseMatrixCSR     LhsM_csr;         // nv × nv  Vᵀ·Lₕ·V
		// Incomplete-Cholesky factor of LhsM_csr — built lazily the
		// first time `use_incomplete_cholesky` is observed true at
		// solve time. Skipped when the flag is off, since
		// factorisation is non-trivial (~28 s at 81k) and we don't
		// want it to delay bind-time invariably.
		curvenet::incomplete_cholesky::IncompleteCholeskyFactor incomplete_cholesky_factor;
		bool                                  incomplete_cholesky_built = false;
		double                                incomplete_cholesky_shift = 0.0;
		int                                   nc          = 0;  // number of sample columns
		std::uint64_t                         source_hash = 0;
		// Warm-start state: the previous frame's solver iterates,
		// fed back into `cg_with_guess` so smooth interactive drags
		// converge in 1-2 CG steps instead of restarting from zero.
		// `prev_*_valid` is false until the first solve completes.
		std::vector<double>                   prev_Fv;          // nv × 9
		std::vector<double>                   prev_Xv;          // nv × 3
		bool                                  prev_solve_valid = false;

		// Direct Delta Mush per-vertex sparse influence list (set when
		// `use_direct_delta_mush` was true at the last bind step).
		// `ddm_influences[v]` is the sparse weight row for vertex v.
		std::vector<std::vector<std::pair<int, double>>> ddm_influences;
		bool                                  ddm_built = false;

		// Rest-pose knot positions per curve. Baked once at the bind step
		// (alongside the cut-mesh) so that subsequent drags can compute
		// per-handle deformation gradients via DeGoes22 §3 scaled-frame
		// ratios — `(BS)·(B̆S̆)⁻¹` — without re-binding. Indexed by
		// (curve_id, knot_idx).
		std::vector<std::vector<curvenet::Vec3>> rest_curve_knots;

		// Rest-pose per-knot tilt (radians around tangent). DeGoes22 §3's
		// tilt control — rotates the local frame's normal/binormal axes
		// without changing the tangent. Baked from Curve3D::get_point_tilt
		// at bind time; Δtilt = posed - rest is composed into F_h at
		// runtime as a rotation around the posed tangent.
		std::vector<std::vector<double>>          rest_curve_tilts;
	};
	mutable RestCache rest_cache;

protected:
	static void _bind_methods();
	void invalidate_cache();

public:
	CurveNetDeformer3D() = default;
	~CurveNetDeformer3D() = default;

	void set_source_path(const NodePath &p_path);
	NodePath get_source_path() const;

	void set_profile_curves(const TypedArray<Curve3D> &p_curves);
	TypedArray<Curve3D> get_profile_curves() const;

	void set_length_tiebreak(double p_v);
	double get_length_tiebreak() const;

	void set_deformation_active(bool p_v);
	bool is_deformation_active() const;

	void set_use_incomplete_cholesky(bool p_v);
	bool get_use_incomplete_cholesky() const;

	void set_use_direct_delta_mush(bool p_v);
	bool get_use_direct_delta_mush() const;
	void set_ddm_top_k(int p_v);
	int  get_ddm_top_k() const;
	void set_ddm_smooth_iters(int p_v);
	int  get_ddm_smooth_iters() const;

	void _ready() override;

	// Recompute the deformed mesh from the current profile_curves and apply
	// it to this MeshInstance3D's mesh. Cheap on subsequent calls because
	// the rest-pose tri->quad fusion + binding is cached.
	void apply_deformation();

	// Introspection for tooling / GDScript: how many faces after tri->quad fusion.
	int get_face_count() const;
	// Returns Mesh::PRIMITIVE_TRIANGLES face count (3 verts) or 4 for quads.
	int get_face_vertex_count(int face_index) const;
	// Evaluate the patch for face `face_index` at parameter (s, t) using the
	// currently authored profile_curves as boundaries (or straight-line cubics
	// if no curves are bound to that face's edges). Returns Vector3.ZERO and
	// prints an error if the face index is out of range or the cache is empty.
	Vector3 evaluate_face(int face_index, double s, double t);
};

} // namespace godot

#endif
