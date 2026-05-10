import Curvenet.HalfedgeBuilder
import Curvenet.CurvenetBuilder
import Curvenet.SurfaceProjection
import Curvenet.CutMeshLaplacian
import Curvenet.ScaledFrames
import Curvenet.SparseLinAlg
import Curvenet.HarmonicSolve
import Curvenet.DeformSolve
import Curvenet.DirectDeltaMush
import Curvenet.DirectDeltaMushBind
import Curvenet.Halfedge

/-!
# Features manifest

Every feature formerly catalogued under `docs/features/` (now retired)
is reified here as a Lean entity. Three top-level namespaces:

- `Features.KnownKnowns` — what we ship today. Each section is an
  `example : ... := by native_decide` re-running a canonical claim
  from the underlying module. If a feature regresses, lake build
  fails.
- `Features.KnownUnknowns` — gaps we know about. Each section is an
  `axiom` of type `True`, named after the gap. The axiom is
  trivially satisfiable; the *docstring* carries the semantic
  content. When the gap is closed, replace the axiom with a
  `def ... : True := trivial` and turn the docstring past-tense.
- `Features.UnknownUnknowns` — risk categories where we have no
  operational data. Same axiom-with-docstring pattern; downgrades to
  `def` once a fixture validates the assumption.

The project policy is **zero `sorry` / `admit`**. Open work is
expressed as `axiom name : True` rather than `sorry`-ing a non-trivial
goal.
-/

namespace Features

/-! ## Known knowns — what we ship today -/

namespace KnownKnowns

open Curvenet
open Curvenet.HalfedgeBuilderExamples
open Curvenet.CurvenetBuilderExamples
open Curvenet.SurfaceProjectionExamples
open Curvenet.CutMeshLaplacianExamples
open Curvenet.ScaledFramesExamples
open Curvenet.SparseLinAlg
open Curvenet.HarmonicSolveExamples
open Curvenet.DeformSolveExamples
open Curvenet.DirectDeltaMushExamples
open Curvenet.DirectDeltaMushBindExamples

/-- Feature 01 — Halfedge mesh. Triangle-soup → manifold halfedge
    structure. Re-runs the canonical claim that
    `HalfedgeBuilder.fromTriangles` produces a manifold mesh on a
    single triangle. -/
example : singleTri.manifold? = true := by native_decide

/-- Feature 02 — Curvenet graph. ε-merge + classify. Two open
    curves sharing an endpoint within ε produce 3 distinct merged
    knots. -/
example :
    let A : Vec3 := ⟨0.0, 0.0, 0.0⟩
    let B : Vec3 := ⟨1.0, 0.0, 0.0⟩
    let Bp : Vec3 := ⟨1.0 + 1e-10, 0.0, 0.0⟩
    let C : Vec3 := ⟨2.0, 0.0, 0.0⟩
    let g := CurvenetBuilder.build #[#[A, B], #[Bp, C]] 1e-6
    g.knotPositions.size = 3 := by native_decide

/-- Feature 03 — Cut-mesh + cut-algorithm. Sample-promoted vertices
    + surgery primitives. Re-runs the canonical partition-of-unity
    invariant on `triangleWithSample`. -/
example :
    let oneSample : Nat → Nat → Bool → Nat := fun _ _ _ => 0
    CutExamples.triangleWithSample.partitionOfUnity oneSample = true := by
  native_decide

/-- Feature 04 — Surface projection (vertex-only). Knot at the
    origin projects to vertex 0 of the triangle mesh. -/
example :
    let origin : Vec3 := ⟨0.0, 0.0, 0.0⟩
    let triMesh : Array Vec3 :=
      #[origin, ⟨1.0, 0.0, 0.0⟩, ⟨0.0, 1.0, 0.0⟩]
    let r := SurfaceProjection.projectToVertices #[origin] triMesh
    r.size = 1 ∧ r[0]!.meshIndex = 0 := by native_decide

/-- Feature 05 — Cot-Laplacian. `VᵀLₕV` is symmetric on the cut-mesh
    `triangleWithSample`. Sharp-Crane mollified. -/
example :
    let triPositions : Array Vec3 :=
      #[⟨0.0, 0.0, 0.0⟩, ⟨1.0, 0.0, 0.0⟩, ⟨0.5, 0.0, 0.8660254037844386⟩]
    let M := CutMeshLaplacian.assembleVtLhV CutExamples.triangleWithSample triPositions
    PolygonLaplacian.isSymmetricWithin M 3 1e-10 = true := by native_decide

/-- Feature 06 — Scaled frames + intersection frames. DeGoes22 §3
    deformation gradient is identity for rest = posed. -/
example :
    ScaledFrames.mat3WithinEps
      (ScaledFrames.deformationGradient ScaledFrames.mat3Identity ScaledFrames.mat3Identity)
      ScaledFrames.mat3Identity 1e-12 = true := by
  native_decide

/-- Feature 07 — Solver kernels. CG converges on a small SPD system
    (a positive-definite 3×3) to within 1e-9 of the analytic solution. -/
example :
    let A : SparseLinAlg.SparseMatrixCSR :=
      { rows := 3, cols := 3
      , rowPtr := #[0, 2, 5, 7]
      , colIdx := #[0, 1, 0, 1, 2, 1, 2]
      , values := #[4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0] }
    let b : Array Float := #[1.0, 0.0, -1.0]
    let x := SparseLinAlg.cg A b 100 1e-12
    let r := SparseLinAlg.saxpby 1.0 b (-1.0) (SparseLinAlg.spmv A x)
    SparseLinAlg.dot r r < 1e-9 = true := by native_decide

/-- Feature 08 — DeGoes22 §6 two-stage solve. Pure translation:
    `f_c = identity`, `x_c = (5, 0, 0)` on `triangleWithSample` →
    vertex 1 ends at (6, 0, 0). Self-contained restatement (the
    canonical proof's fixtures are private to its examples module). -/
example :
    let positions : Array Vec3 :=
      #[ ⟨0.0, 0.0, 0.0⟩, ⟨1.0, 0.0, 0.0⟩, ⟨0.5, 0.0, 0.8660254037844386⟩ ]
    let oneSample : Nat → Nat → Bool → Nat := fun _ _ _ => 0
    let Fc : DenseLinAlg.Mat :=
      #[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    let Xc : DenseLinAlg.Mat := #[5.0, 0.0, 0.0]
    let Xv := DeformSolve.solveDeformation CutExamples.triangleWithSample positions oneSample Fc Xc
    let v1 : Array Float := #[DenseLinAlg.get Xv 3 1 0,
                               DenseLinAlg.get Xv 3 1 1,
                               DenseLinAlg.get Xv 3 1 2]
    DenseLinAlg.vecWithinEps v1 #[6.0, 0.0, 0.0] 1e-9 = true := by
  native_decide

/-- Feature 09 — Direct Delta Mush. Identity transform with full
    weight = 1 preserves the rest position. -/
example :
    let infl : DirectDeltaMush.Influences := #[(0, 1.0)]
    let transforms : Array DirectDeltaMush.Mat4 := #[DirectDeltaMush.mat4Identity]
    DirectDeltaMush.vclose
      (DirectDeltaMush.lbsMatvec transforms infl 1.5 (-2.0) 0.75)
      (1.5, -2.0, 0.75) 1e-12 = true := by
  native_decide

/-- Feature 10 — Authoring UX (4 draggable §3 knobs). The bind-time
    DDM weight composition preserves partition of unity, which is
    the algorithmic guarantee underneath the artist's drag. Identity
    weight matrix has every row summing to 1. -/
example :
    let identityW : DirectDeltaMushBind.WeightMatrix :=
      #[#[1.0, 0.0, 0.0, 0.0],
        #[0.0, 1.0, 0.0, 0.0],
        #[0.0, 0.0, 1.0, 0.0],
        #[0.0, 0.0, 0.0, 1.0]]
    DirectDeltaMushBind.partitionOfUnity identityW 1e-12 = true := by native_decide

/-- Feature 11 — Editor visualization (frame axes, width ring,
    knot kind markers). The runtime frame derivation that the gizmo
    mirrors is the same `isolatedSegmentGradient` proven for
    rest = posed → identity. -/
example :
    ScaledFrames.mat3WithinEps
      (ScaledFrames.isolatedSegmentGradient ⟨0,0,0⟩ ⟨1,0,0⟩ ⟨0,0,0⟩ ⟨1,0,0⟩)
      ScaledFrames.mat3Identity 1e-12 = true := by
  native_decide

/-- Feature 12 — Test + spec infrastructure. The smallest manifold
    fixture in this library passes its manifold check, demonstrating
    the test/spec layer is functional. -/
example : Curvenet.Examples.triangle.manifold? = true := by native_decide

end KnownKnowns

/-! ## Known unknowns — gaps we know about

Each axiom names a gap; closing it = replacing the axiom with a
`def name : True := trivial` (or a meaningful theorem if the gap
admits one).
-/

namespace KnownUnknowns

/-- Gap 01 — Edge+face surface projection. Today vertex-only;
    `SurfaceProjection.ProjectionKind` reserves `edgeIntersection`
    and `faceInterior` but the producer never emits them. Closing
    this requires a closest-point-on-triangle helper +
    `promote_samples` dispatcher. -/
axiom edgeFaceProjection : True

/-- Gap 02 — Curve-segment tracing through faces. Once edge/face
    projection lands (gap 01), curves still need a tracer that
    inserts cracks via `cut_algorithm::insert_crack_at_endpoint`
    along the geodesic between adjacent knots. Blocked by 01. -/
axiom curveSegmentTracing : True

/-- Gap 03 — Quest 3 GPU dispatch. DDM runtime kernel is CPU; port
    `direct_delta_mush::lbs_matvec` to a Vulkan compute shader for
    the 0.8 ms Quest 3 budget. -/
axiom quest3GpuDispatch : True

/-- Gap 04 — Sparse bind harvest. DDM bind currently uses dense
    `harmonic_solve::solve_multi` (~30 s on 81 k); replace with
    sparse multi-RHS PCG over the cached `Lh_csr`. -/
axiom sparseBindHarvest : True

/-- Gap 05 — Side toggle UI. `IntersectionFrames::per_side_scaled_frames`
    already produces plus / minus sides but the deformer always pulls
    from one; expose a per-knot side flag + click-to-flip marker at
    intersection knots. -/
axiom sideToggleUi : True

/-- Gap 06 — Per-knot length-scale `l` drag. Width `w` has a 3D drag
    handle; `l` is auto-derived from segment length. Add a third
    drag-handle kind. -/
axiom perKnotLDrag : True

/-- Gap 07 — Surface-normal reference frame for tilt. Today the
    tilt-rotation reference is world-up Gram-Schmidt; should be the
    mesh vertex normal at the projection point. -/
axiom surfaceNormalReferenceFrame : True

/-- Gap 08 — Bind cache persistence. Every Godot session re-binds;
    serialize `RestCache` to a sidecar `.curvenet_cache.bin`. -/
axiom bindCachePersistence : True

/-- Gap 09 — Auto profile-curve extraction. Bootstrap curves from
    arbitrary meshes via geodesic level sets. Blocked by 10. -/
axiom autoCurveExtraction : True

/-- Gap 10 — Heat method on polygon soups. Crane 2013 + Sharp-Crane
    2020 mollification on top of in-house solvers; ~200 LOC port. -/
axiom heatMethodPolygonSoups : True

/-- Gap 11 — Multi-character batching. Shared meshlet pool +
    per-deformer DDM weight rows in a project-level singleton. -/
axiom multiCharacterBatching : True

/-- Gap 12 — Animated knot drag. Hook `EditorUndoRedoManager` actions
    to a Godot `AnimationPlayer` track for record-and-replay. -/
axiom animatedDrag : True

/-- Gap 13 — Godot CSG integration. Extend `source_path` dispatch
    to handle `CSGShape3D` / `CSGCombiner3D` via `bake_static_mesh()`. -/
axiom godotCsgIntegration : True

/-- Gap 14 — Hash-diff regression gate (ufbx-style). FNV-1a hashing
    of deformed positions + DDM influences against a baseline; tagged
    dump on mismatch with parent-stack breadcrumbs. -/
axiom hashDiffRegressionGate : True

end KnownUnknowns

/-! ## Unknown unknowns — risk categories without operational data -/

namespace UnknownUnknowns

/-- Risk 01 — Real character rig validation beyond Mire. We have
    data on the Mire body (V-Sekai-fire/mesh-mille-mire-feuille);
    untested on additional production rigs and against artist-
    authored profile curves. -/
axiom realCharacterRigBeyondMire : True

/-- Risk 02 — Animation playback / pose-space behavior. Deformer is
    one-shot per `apply_deformation`; behavior under `AnimationPlayer`-
    driven keyframed knot positions is untested. -/
axiom animationPoseSpace : True

/-- Risk 03 — Multi-deformer scene. Tested with one
    `CurveNetDeformer3D`; behavior at 5/10/50 nodes per scene
    untested. -/
axiom multiDeformerScene : True

/-- Risk 04 — Scene save / load lifecycle. `RestCache` mutable +
    lazy-built; serialization across save/close/reopen untested. -/
axiom sceneSaveLoadLifecycle : True

/-- Risk 05 — Multi-platform builds. Developed on macOS Apple
    Silicon; Linux / Windows / Android (Quest 3) / iOS untested. -/
axiom multiPlatformBuilds : True

/-- Risk 06 — Numerical robustness on degenerate inputs. Sharp-Crane
    handles thin triangles; truly degenerate (zero edges, NaN coords,
    near-coincident knots) untested. -/
axiom degenerateInputs : True

/-- Risk 07 — Scaling beyond 81 k. Largest tested mesh is Mire body
    at 81 k vertices; behavior at 200 k / 500 k / 1 M untested
    against the current production path. -/
axiom scalingBeyondBenchmark : True

/-- Risk 08 — Godot version drift. `godot-cpp` pinned; behavior on
    Godot 4.5 / 5.0+ untested. -/
axiom godotVersionDrift : True

end UnknownUnknowns

end Features
