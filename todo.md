# todo

## Where I am

- On `main` at `380224e` (latest merge: PR #21).
- All test suites green locally:
  - `cd lean && lake build` — 110 jobs, every `native_decide` proof accepts.
  - `make -C tests/safetensors test` — 9/9 vectors.
  - `make -C tests/slang_validate test` — 6/6 kernels.
- No outstanding branches; everything pushed and merged.

## Shipped today

| PR | Subject |
|----|---------|
| #14 (closed via #17 split) | safetensors → glTF pipeline for Mire |
| #17 | Split Z-up→Y-up into a proper kernel pass (adapter → kernel → decoder) |
| #18 | cuJSON `checkAscii` port — first slang_validate test |
| #19 | Smallest working Profile-Curves integration (Bézier → solver → .glb) |
| #20 | Switch demo to rotation case + ping-pong morph animation |
| #21 | **MVP**: slang_validate `direct_delta_mush` — bit-exact on rotation case |

## Gall's-Law next steps (smallest → largest)

Each replaces one component on the working baseline. Pick the top one to resume.

1. **Multi-handle DDM test** — extend `direct_delta_mush_test.cpp` to 2 handles per vertex with non-trivial weight blending (e.g. 70/30 split). Same harness shape; reference adds a Σ over handles. ~60 LOC new. Validates the CSR matvec at non-trivial weight density.

2. **Harmonic CG dispatch** — drive `DeformSolve.solveDeformation` through slangc-emitted `spmv` + `dot_reduce` + `saxpby` in a CG loop. Bit-diff against the Lean reference. ~400 LOC: new orchestrator + per-iteration buffer plumbing. This is the *first* GPU dispatch in the project's history. Heavy lift.

3. **Real mesh past `triangleWithSample`** — start with a tetrahedron (4 verts, 4 faces); next a unit cube subdivided into 12 tris. Requires authoring DDM weights manually (or via the unwired bind-time harvest path).

4. **Bind-time weight harvest** — exercise `DirectDeltaMushBind.{smoothWeights, sparsifyTopK}` end-to-end on a real mesh. Currently both are validated only on a 4-vertex path graph in `DirectDeltaMushBind.lean:67–200`.

5. **Mire integration** — populate the `EXT_structural_metadata` `Curvenet / Segment / Knot` classes on the Mire `.glb` output with a real curvenet rig. Requires authoring (or auto-generating) a curvenet that drives the 117-bone Mire armature. Largest scope; defer until 1–4 are stable.

## Open context / gotchas

- **Lean 4 reserved word `meta`** — bit me once in `SafetensorsZupToYup.lean`. Rename to `newMeta` or similar in any new code that needs a local `meta` binding.
- **CI bot-comment apostrophes** — `.github/workflows/ci.yml` heredoc trips on `'` in the body text. Use straight prose without contractions in any future comment-body changes.
- **slangc cpp target has no `InterlockedOr`** — if porting a cuJSON kernel that uses atomics, restructure to per-thread outputs + host-side reduction (see `Curvenet.SlangCodegen.CheckAscii.lean` for the pattern).
- **DDM kernel has no bounds check** — when adding more DDM test variants, pad `inflStart` to `GROUP_SIZE + 1` with trailing entries = `nnz_total` so out-of-range threads see an empty influence range (see `direct_delta_mush_test.cpp:64–69`).
- **`.claude/scheduled_tasks.lock`** — harness-managed; always shows as deleted/modified in `git status`. Stash before rebase, drop after; never stage.
- **V-Sekai-fire squash workflow** — local `git reset --soft origin/main` to one commit + `gh pr merge --merge --auto`. **Banned**: `gh pr merge --squash` (server-side squash).

## Where everything lives

- Spec (DeGoes22 §3 + §4): `lean/Curvenet/{CurveInterp,ScaledFrames,SegmentGradient,IntersectionFrames,CurvenetBuilder,CutAlgorithm,CutMeshLaplacian,HarmonicSolve,DeformSolve,DirectDeltaMush,DirectDeltaMushBind}.lean`
- Slang AST + emitter: `lean/Curvenet/SlangCodegen/` (33 kernels; +`CheckAscii` for cuJSON)
- Smallest end-to-end demo: `lean/Curvenet/EndToEndExample.lean` + `lean/ProfileCurvesDemo.lean`
- Test harnesses: `tests/slang_validate/*.cpp` (6 kernels), `tests/safetensors/{gen_vectors,run_tests}.py` (9 vectors)
- CI: `.github/workflows/ci.yml` — single `validate` job on macos-14, posts a nightly.link comment with the demo .glb on every PR
