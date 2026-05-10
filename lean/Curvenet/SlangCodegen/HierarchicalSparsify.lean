import LeanSlang

/-!
# `Curvenet.SlangCodegen.HierarchicalSparsify` — coarse-vertex restriction

Slang re-port of the per-coarse-vertex restriction pass at the
heart of `src/curvenet/hierarchical_sparsify.h`. The full algorithm
is a multi-level coarsening pipeline (level-0 fine mesh →
intermediate aggregated levels → coarsest); this module models one
level's restriction step.

For each coarse vertex `c`, sum the values at the fine vertices
that map to `c`:
  coarse[c] = Σ over fine i with parent[i] = c    fine[i]

Bindings:
  0 — `ConstantBuffer<HSparsifyParams> { uint num_coarse; uint num_fine; }`
  1 — `StructuredBuffer<int>   parent`     (length num_fine, parent[i] = c)
  2 — `StructuredBuffer<float> fine_vals`  (length num_fine)
  3 — `RWStructuredBuffer<float> coarse_vals` (length num_coarse)

Per-thread work: linear scan over fine vertices to find children of
the active coarse vertex. The production form uses a CSR-by-coarse
mapping; this kernel models the per-coarse summation shape.
-/

namespace Curvenet.SlangCodegen.HierarchicalSparsify

open LeanSlang

private def floatTy : SlangType := .scalar .float
private def intTy   : SlangType := .scalar .int
private def uintTy  : SlangType := .scalar .uint

def shader : SlangShaderModule :=
  { structs :=
      [ { name := "HSparsifyParams"
        , fields :=
            [ ⟨"num_coarse", .scalar .uint, Semantic.none, none, none, .qIn⟩
            , ⟨"num_fine",   .scalar .uint, Semantic.none, none, none, .qIn⟩ ] } ]
  , globals :=
      [ ⟨"params",      .const "HSparsifyParams", Semantic.none, some 0, some 0, .qIn⟩
      , ⟨"parent",      .roBuf intTy,             Semantic.none, some 1, some 0, .qIn⟩
      , ⟨"fine_vals",   .roBuf floatTy,           Semantic.none, some 2, some 0, .qIn⟩
      , ⟨"coarse_vals", .rwBuf floatTy,           Semantic.none, some 3, some 0, .qIn⟩ ]
  , functions := [{
      attrs  := [.shaderCompute, .numthreads 256 1 1]
      name   := "main"
      params := [⟨"tid", .vec .uint 3, .svDispatchThreadId, none, none, .qIn⟩]
      body   :=
        [ .declInit uintTy "c" (.member (.var "tid") "x")
        , .ifNoElse (.bin ">=" (.var "c") (.member (.var "params") "num_coarse"))
            [ .ret none ]
        , .declInit floatTy "acc" (.litFloat 0.0)
        , .forCount "i" (.litUint 0) (.member (.var "params") "num_fine")
            [ .ifNoElse
                (.bin "==" (.call "uint" [.index (.var "parent") (.var "i")])
                            (.var "c"))
                [ .assign (.var "acc")
                    (.bin "+" (.var "acc")
                              (.index (.var "fine_vals") (.var "i"))) ] ]
        , .assign (.index (.var "coarse_vals") (.var "c")) (.var "acc")
        , .ret none ] }] }

example : shader.entryPointName = "main" := by native_decide

end Curvenet.SlangCodegen.HierarchicalSparsify
