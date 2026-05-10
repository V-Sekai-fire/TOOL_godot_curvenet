# TOOL_godot_curvenet

Lean 4 implementation of Pixar's **Profile Curves** character
articulation algorithm (de Goes, Sheffler, Fleischer, *SIGGRAPH 2022*).
Lean is the source of truth — every algorithm is specified as Lean
`def`s with `native_decide` proofs of the load-bearing invariants on
small instances. From those Lean specs, the project also generates
**Slang** compute shaders via [V-Sekai-fire/lean-slang][1] and
round-trips each one through three validation layers: text-pin
fixtures, `slangc -target spirv`, and a slangc-cpp + clang harness
that proves spec ≡ kernel on real inputs.

[1]: https://github.com/V-Sekai-fire/lean-slang
[2]: https://github.com/DevPrice/godot-slang
[3]: https://github.com/shader-slang/slang
[4]: https://github.com/V-Sekai-fire/TOOL_godot_curvenet/actions

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

The Lean package under `lean/` defines the algorithm spec
(`Curvenet.*`) and the Slang codegen (`Curvenet.SlangCodegen.*`).
Each codegen module exports a `shader : SlangShaderModule`, a pinned
`expected : String`, and two `native_decide` lemmas: one for emit
equality and one for the entry-point name. `Curvenet.SlangCodegen.Common`
holds the shared building blocks (type aliases, mat3 storage helpers,
3×3 inverse, cot weight, four Knuth/Dekker EFT primitives).

The Lake `extern_lib axpy_native` compiles the slangc-cpp output of
`axpy.slang` plus a C-ABI shim into a static library; `Curvenet.SlangValidate.Axpy`
declares the C entry point as a Lean `@[extern]`, and `lake exe axpy_validate`
runs three fixtures comparing the kernel output to a pure-Lean
`axpySpec`. The pattern is mechanical to extend to other kernels.

The standalone C++ test harness under `tests/slang_validate/` runs
the same end-to-end check (slangc-cpp → clang → run on hand-rolled
inputs → assert) for `axpy`, `saxpby`, and `jacobi` without the Lean
FFI. `bin/slangc` wraps the upstream Slang toolchain that
`misc/install-slang.sh` downloads to `bin/.slang/`.

## Build

```sh
cd lean && lake build                 # 100+ jobs, including every
                                      # native_decide pin (algorithm
                                      # invariants + emission text)

misc/install-slang.sh                 # one-time: fetches the slangc
                                      # toolchain into bin/.slang/
                                      # (~200MB, gitignored)

cd lean && lake exe emit_shaders /tmp/out
                                      # dump all 32 .slang files to disk

bin/slangc -target spirv -profile sm_6_5 -stage compute \
  -entry main -o /tmp/axpy.spv /tmp/out/axpy.slang
                                      # round-trip emitted text through
                                      # the upstream Slang compiler

make -C tests/slang_validate test     # real-input checks: axpy, saxpby,
                                      # jacobi via slangc -target cpp +
                                      # clang + run

cd lean && lake exe axpy_validate     # spec ≡ slangc-emitted-kernel
                                      # equivalence check on three
                                      # fixtures via the FFI bridge
```

The Lean side is self-contained — no C++ build needed for `lake build`
to succeed. Slangc validation requires `bin/.slang/`. Real-input + FFI
equivalence additionally need a host C++ compiler.

## Validation pipeline

Every Slang kernel is gated by four layers:

| Layer | What it catches | Coverage |
|---|---|---|
| `native_decide` on emission text | LeanSlang pretty-printer drift | 32 / 32 |
| `slangc -target spirv` round-trip | Slang syntactic / semantic errors, SPIR-V codegen failures | 32 / 32 |
| `slangc -target metal` + `xcrun metal` + `xcrun metallib` | MSL-specific issues + AOT compile to Apple `.metallib` | 31 / 32 (skips `direct_delta_mush` — slangc Metal codegen bug; SPIR-V path is fine) |
| `slangc -target cpp` + clang + run on known inputs | Wrong arg order, swapped binding, off-by-one, anything that compiles cleanly but computes the wrong thing | 3 / 32 (axpy, saxpby, jacobi; pattern is per-kernel ~50 LOC) |

The Lean `extern_lib axpy_native` builds atop the cpp target: the
slangc-emitted CPU function is wrapped by `lean/native/axpy_shim.cpp`
into a Lean-callable extern, then `lake exe axpy_validate` runs three
fixtures comparing the kernel output to a pure-Lean `axpySpec`. This
is the cleanest "spec ≡ kernel" gate — extending it to other kernels
is mechanical (~50 LOC of shim + ~30 LOC of harness per kernel).

## Layer mapping

| §  | Lean spec | Slang kernel |
|----|-----------|--------------|
| §3 curvenet | `Curvenet.{IntersectionFrames, CurveInterp, SegmentGradient, ScaledFrames, CurvenetBuilder}` | `Curvenet.SlangCodegen.{ScaledFrames, SegmentGradient, IntersectionFrames, CurveInterp, CurvenetBuilder}` |
| §4.1 cut-mesh | `Curvenet.{Halfedge, HalfedgeBuilder, CutMesh, CutAlgorithm, SurfaceProjection}` | `Curvenet.SlangCodegen.{Halfedge, HalfedgeBuilder, CutMesh, CutAlgorithm, SurfaceProjection}` |
| §4.2 discretisation | `Curvenet.{PolygonLaplacian, RobustLaplacian, CutMeshLaplacian}` | `Curvenet.SlangCodegen.{PolygonLaplacian, RobustLaplacian, CutMeshLaplacian}` |
| solver kernels | `Curvenet.{DenseLinAlg, SparseLinAlg, IncompleteCholesky, HierarchicalSparsifyCompensate, GraphColoring}` | `Curvenet.SlangCodegen.{DenseLinAlg, SparseLinAlg, IncompleteCholesky, HierarchicalSparsify}` plus the BLAS / preconditioner kernels (`Axpy`, `AxpyMulti`, `Saxpby`, `SaxpbyMulti`, `Jacobi`, `JacobiMulti`, `Spmv`, `SpmvMulti`, `DotReduce`, `DotReduceMulti`, `SgsColor`) |
| §4.3 solve | `Curvenet.{HarmonicSolve, DeformSolve}` | `Curvenet.SlangCodegen.{HarmonicSolve, DeformSolve}` |
| runtime kernel | `Curvenet.{DirectDeltaMush, DirectDeltaMushBind}` | `Curvenet.SlangCodegen.DirectDeltaMush` |
| shared helpers | — | `Curvenet.SlangCodegen.Common` (type aliases, mat3 storage, 3×3 inverse, cot weight, four Knuth/Dekker EFT primitives) |

## Continuous integration

[`.github/workflows/ci.yml`][4] runs two jobs on every push and PR:

- **`lean-and-slang`** on `ubuntu-latest` — `lake build` (every
  `native_decide`), `slangc -target spirv` round-trip on all 32
  kernels (asserts SPIR-V magic `0x07230203` + size > 100 bytes),
  the slang_validate harness, and the `axpy_validate` equivalence
  check.
- **`metal`** on `macos-14` — same as above plus the full Slang →
  `.metal` source → `xcrun metal` (.air) → `xcrun metallib` chain
  (asserts `MTLB` magic). `macos-13` Intel runners are intentionally
  avoided because they lack GPU paravirtualization.

The `main` branch is protected: every change goes through a PR with
both checks green before it can land. `enforce_admins: true` — admins
included, no bypass.

## Slang language reference

The emitted shaders target the [Slang shader language][3]. Language
spec is at <https://github.com/shader-slang/spec>. The
`Curvenet.SlangCodegen.*` modules build a `LeanSlang.SlangShaderModule`
in pure Lean and the LeanSlang pretty-printer emits `slangc`-acceptable
text from it; AST extensions (e.g., `groupshared`, `precise`, ternary)
go upstream into V-Sekai-fire/lean-slang.

## Acknowledgements

Bibliographic citations live in `references.bib` (35 entries). The
load-bearing references for the algorithm and the runtime are:

- `DeGoes2022Curvenet` — de Goes, Sheffler, Fleischer, *Character
  Articulation through Profile Curves*, SIGGRAPH 2022.
- `Nguyen2023Elemental` — Nguyen et al, *Shaping the Elements:
  Curvenet Animation Controls in Pixar's Elemental*, SIGGRAPH 2023
  Talks.
- `LeLewis2019DirectDeltaMush` — Le & Lewis, *Direct Delta Mush
  Skinning and Variants*, SIGGRAPH 2019 — the runtime kernel that
  bakes the §4.3 harmonic solve and replays it as a sparse linear
  combination per vertex.
- `KrishnanFattalSzeliski2013HSC` — Krishnan, Fattal, Szeliski,
  *Efficient Preconditioning of Laplacian Matrices for Computer
  Graphics*, SIGGRAPH 2013 — the HSC preconditioner that drives the
  §4.3 inner CG iterations.
