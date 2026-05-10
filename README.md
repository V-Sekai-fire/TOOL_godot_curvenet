# TOOL_godot_curvenet

Lean 4 implementation of Pixar's **Profile Curves** character
articulation algorithm (de Goes, Sheffler, Fleischer, *SIGGRAPH 2022*).
Lean is the source of truth — every algorithm is specified as Lean
`def`s with `native_decide` proofs of the load-bearing invariants on
small instances. From those Lean specs, the project also generates
**Slang** compute shaders via [V-Sekai-fire/lean-slang][1] that pass
upstream `slangc -target spirv` to produce SPIR-V for runtime
dispatch through engines such as [DevPrice/godot-slang][2].

[1]: https://github.com/V-Sekai-fire/lean-slang
[2]: https://github.com/DevPrice/godot-slang

## Why

DeGoes22 deforms a surface by solving a two-stage Poisson system on a
**cut-mesh** that splits triangles wherever a curvenet of cubic Bézier
curves crosses them. Each curvenet knot carries scaled per-side frames
that drive the deformation gradient field; the resulting interpolation
respects silhouettes and detail in a way cage- and skinning-based
deformers don't. Compared to tris-to-quads + Coons-patch:

* runs on the input triangle mesh as-is — no quad-fusion pre-pass
* handles intersections, anchors, and shared knots in arbitrary
  curvenet topologies (panda-style multi-curve rigs, not just one
  closed loop)
* propagates deformation globally via a sparse harmonic solve, so a
  single moved knot smoothly affects the whole region

## Architecture

```
lean/Curvenet/                      Lean 4 spec + native_decide proofs
  Vec3.lean, Halfedge.lean,
    HalfedgeBuilder.lean,
    CurvenetBuilder.lean,
    SurfaceProjection.lean,
    PolygonLaplacian.lean,
    RobustLaplacian.lean,
    CutMesh.lean,
    CutAlgorithm.lean,
    CutMeshLaplacian.lean,
    ScaledFrames.lean,
    IntersectionFrames.lean,
    CurveInterp.lean,
    SegmentGradient.lean,
    DirectDeltaMush.lean,
    DirectDeltaMushBind.lean,
    DenseLinAlg.lean,
    SparseLinAlg.lean,
    IncompleteCholesky.lean,
    HierarchicalSparsifyCompensate.lean,
    GraphColoring.lean,
    HarmonicSolve.lean,
    DeformSolve.lean              The 23-module algorithm spec.

  SlangCodegen/                     One Slang shader per algorithm,
    Vec3.lean, Axpy.lean,           emitted via LeanSlang. Each module
    AxpyMulti.lean, Saxpby.lean,    exports `shader : SlangShaderModule`
    SaxpbyMulti.lean, Jacobi.lean,  + a pinned `expected : String`.
    JacobiMulti.lean, Spmv.lean,
    SpmvMulti.lean, DotReduce.lean,
    DotReduceMulti.lean,
    SgsColor.lean,
    PolygonLaplacian.lean,
    RobustLaplacian.lean,
    ScaledFrames.lean,
    SegmentGradient.lean,
    IntersectionFrames.lean,
    CurveInterp.lean,
    Halfedge.lean,
    HalfedgeBuilder.lean,
    CutMesh.lean,
    CutAlgorithm.lean,
    SurfaceProjection.lean,
    CurvenetBuilder.lean,
    IncompleteCholesky.lean,
    HierarchicalSparsify.lean,
    CutMeshLaplacian.lean,
    DenseLinAlg.lean,
    SparseLinAlg.lean,
    HarmonicSolve.lean,
    DeformSolve.lean,
    DirectDeltaMush.lean

EmitShaders.lean                    Lake exe: dumps every Slang
                                    shader to a directory for slangc.
bin/slangc                          Shim around bin/.slang/bin/slangc
                                    (gitignored ~200MB toolchain).
misc/install-slang.sh               Refetcher.
```

## Build

```sh
cd lean && lake build              # checks every native_decide proof
                                   # + the Slang fixture pin per kernel

lake exe emit_shaders /tmp/out     # dump every Slang shader to disk
                                   # (used as input to slangc)

bin/slangc -target spirv -profile sm_6_5 -stage compute \
  -entry main -o /tmp/foo.spv /tmp/out/axpy.slang
                                   # round-trip emitted text through
                                   # the upstream Slang compiler

misc/install-slang.sh              # one-time: fetches the slangc
                                   # toolchain into bin/.slang/
```

The Lean side is self-contained — no C++ build needed. The kernel
emissions can be slangc-validated locally; consumption (e.g.
`godot-slang`'s `ComputeShaderTask`) is a separate project's
concern.

## Layer mapping

| §  | Lean spec | Slang kernel |
|----|-----------|--------------|
| §3 curvenet | `Curvenet.{IntersectionFrames, CurveInterp, SegmentGradient, ScaledFrames, CurvenetBuilder}` | `Curvenet.SlangCodegen.{ScaledFrames, SegmentGradient, IntersectionFrames, CurveInterp, CurvenetBuilder}` |
| §4.1 cut-mesh | `Curvenet.{Halfedge, HalfedgeBuilder, CutMesh, CutAlgorithm, SurfaceProjection}` | `Curvenet.SlangCodegen.{Halfedge, HalfedgeBuilder, CutMesh, CutAlgorithm, SurfaceProjection}` |
| §4.2 discretisation | `Curvenet.{PolygonLaplacian, RobustLaplacian, CutMeshLaplacian}` | `Curvenet.SlangCodegen.{PolygonLaplacian, RobustLaplacian, CutMeshLaplacian}` |
| solver kernels | `Curvenet.{DenseLinAlg, SparseLinAlg, IncompleteCholesky, HierarchicalSparsifyCompensate, GraphColoring}` | `Curvenet.SlangCodegen.{DenseLinAlg, SparseLinAlg, IncompleteCholesky, HierarchicalSparsify}` plus the BLAS / preconditioner kernels (`Axpy`, `AxpyMulti`, `Saxpby`, `SaxpbyMulti`, `Jacobi`, `JacobiMulti`, `Spmv`, `SpmvMulti`, `DotReduce`, `DotReduceMulti`, `SgsColor`) |
| §4.3 solve | `Curvenet.{HarmonicSolve, DeformSolve}` | `Curvenet.SlangCodegen.{HarmonicSolve, DeformSolve}` |
| runtime kernel | `Curvenet.{DirectDeltaMush, DirectDeltaMushBind}` | `Curvenet.SlangCodegen.DirectDeltaMush` |

Total Slang surface: **32 kernels**, every one round-tripping through
`slangc -target spirv` to valid SPIR-V.

## Acknowledgements

Bibliographic citations live in `references.bib`. The two load-bearing
papers are:

- `DeGoes2022Curvenet` — de Goes, Sheffler, Fleischer, *Character
  Articulation through Profile Curves*, SIGGRAPH 2022.
- `Nguyen2023Elemental` — Nguyen et al, *Shaping the Elements:
  Curvenet Animation Controls in Pixar's Elemental*, SIGGRAPH 2023
  Talks.
