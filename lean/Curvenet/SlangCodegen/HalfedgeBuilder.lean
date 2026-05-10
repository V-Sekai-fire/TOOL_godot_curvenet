import LeanSlang

/-!
# `Curvenet.SlangCodegen.HalfedgeBuilder` — twin-pair resolution

Slang re-port of the twin-resolution step in
`src/curvenet/halfedge_builder.h`. Given face-corner halfedges
indexed `0..3*F-1` (one per triangle corner) and their (source, target)
vertex pairs, resolve each halfedge's twin by finding the unique
opposite-direction halfedge with (source = our.target, target = our.source).

Per-thread work: linear search through the source/target arrays for
the matching reverse pair. O(N²) total but parallel; the host can
swap in a hash-grid for production at the cost of an additional
build pass.

Bindings (set 0):
  0 — `ConstantBuffer<HBuilderParams> { uint he_count; }`
  1 — `StructuredBuffer<int> sources`
  2 — `StructuredBuffer<int> targets`
  3 — `RWStructuredBuffer<int> twins`  (output: -1 if no twin)
-/

namespace Curvenet.SlangCodegen.HalfedgeBuilder

open LeanSlang

private def intTy  : SlangType := .scalar .int
private def uintTy : SlangType := .scalar .uint

def shader : SlangShaderModule :=
  { structs :=
      [ { name := "HBuilderParams"
        , fields := [⟨"he_count", .scalar .uint, Semantic.none, none, none, .qIn⟩] } ]
  , globals :=
      [ ⟨"params",  .const "HBuilderParams", Semantic.none, some 0, some 0, .qIn⟩
      , ⟨"sources", .roBuf intTy,            Semantic.none, some 1, some 0, .qIn⟩
      , ⟨"targets", .roBuf intTy,            Semantic.none, some 2, some 0, .qIn⟩
      , ⟨"twins",   .rwBuf intTy,            Semantic.none, some 3, some 0, .qIn⟩ ]
  , functions := [{
      attrs  := [.shaderCompute, .numthreads 256 1 1]
      name   := "main"
      params := [⟨"tid", .vec .uint 3, .svDispatchThreadId, none, none, .qIn⟩]
      body   :=
        [ .declInit uintTy "i" (.member (.var "tid") "x")
        , .ifNoElse (.bin ">=" (.var "i") (.member (.var "params") "he_count"))
            [ .ret none ]
        , .declInit intTy "src_i" (.index (.var "sources") (.var "i"))
        , .declInit intTy "tgt_i" (.index (.var "targets") (.var "i"))
        , .declInit intTy "found" (.un "-" (.litUint 1))
        , .forCount "j" (.litUint 0) (.member (.var "params") "he_count")
            [ .ifNoElse
                (.bin "&&"
                  (.bin "==" (.index (.var "sources") (.var "j")) (.var "tgt_i"))
                  (.bin "==" (.index (.var "targets") (.var "j")) (.var "src_i")))
                [ .assign (.var "found") (.call "int" [.var "j"]) ] ]
        , .assign (.index (.var "twins") (.var "i")) (.var "found")
        , .ret none ] }] }

def expected : String :=
"struct HBuilderParams {
  uint he_count;
};

[[vk::binding(0, 0)]]
ConstantBuffer<HBuilderParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<int> sources;
[[vk::binding(2, 0)]]
StructuredBuffer<int> targets;
[[vk::binding(3, 0)]]
RWStructuredBuffer<int> twins;

[shader(\"compute\")] [numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint i = tid.x;
  if ((i >= params.he_count)) {
    return;
  }
  int src_i = sources[i];
  int tgt_i = targets[i];
  int found = (-1u);
  for (uint j = 0u; j < params.he_count; ++j) {
    if (((sources[j] == tgt_i) && (targets[j] == src_i))) {
      found = int(j);
    }
  }
  twins[i] = found;
  return;
}"

example : LeanSlang.emit shader = expected := by native_decide

example : shader.entryPointName = "main" := by native_decide

end Curvenet.SlangCodegen.HalfedgeBuilder
