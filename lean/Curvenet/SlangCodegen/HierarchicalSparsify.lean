import LeanSlang
import Curvenet.SlangCodegen.Common

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
open Curvenet.SlangCodegen.Common


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

def expected : String :=
"struct HSparsifyParams {
  uint num_coarse;
  uint num_fine;
};

[[vk::binding(0, 0)]]
ConstantBuffer<HSparsifyParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<int> parent;
[[vk::binding(2, 0)]]
StructuredBuffer<float> fine_vals;
[[vk::binding(3, 0)]]
RWStructuredBuffer<float> coarse_vals;

[shader(\"compute\")] [numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint c = tid.x;
  if ((c >= params.num_coarse)) {
    return;
  }
  float acc = 0.000000;
  for (uint i = 0u; i < params.num_fine; ++i) {
    if ((uint(parent[i]) == c)) {
      acc = (acc + fine_vals[i]);
    }
  }
  coarse_vals[c] = acc;
  return;
}"

example : LeanSlang.emit shader = expected := by native_decide

example : shader.entryPointName = "main" := by native_decide

end Curvenet.SlangCodegen.HierarchicalSparsify
