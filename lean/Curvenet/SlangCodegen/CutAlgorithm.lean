import LeanSlang
import Curvenet.SlangCodegen.Common

/-!
# `Curvenet.SlangCodegen.CutAlgorithm` — per-face cut classification

Slang re-port of the classification step in
`src/curvenet/cut_algorithm.h`: each triangle gets a label based on
how many of its three edges are crossed by some cut curve. The
labels match DeGoes22 §4.1's vertex-promotion / edge-split cases.

  0 — uncut       (no edges crossed)
  1 — single cut  (1 edge crossed; promote one vertex)
  2 — double cut  (2 edges crossed; split + retriangulate)
  3 — full cut    (all 3 edges crossed; rare, robust path)

Bindings:
  0 — `ConstantBuffer<CutAlgoParams> { uint num_faces; }`
  1 — `StructuredBuffer<uint> edge_crossed` (3 entries per face: 0/1)
  2 — `RWStructuredBuffer<uint> labels`     (length num_faces, 0..3)
-/

namespace Curvenet.SlangCodegen.CutAlgorithm

open LeanSlang
open Curvenet.SlangCodegen.Common


def shader : SlangShaderModule :=
  { structs :=
      [ { name := "CutAlgoParams"
        , fields := [⟨"num_faces", .scalar .uint, Semantic.none, none, none, .qIn⟩] } ]
  , globals :=
      [ ⟨"params",       .const "CutAlgoParams", Semantic.none, some 0, some 0, .qIn⟩
      , ⟨"edge_crossed", .roBuf uintTy,          Semantic.none, some 1, some 0, .qIn⟩
      , ⟨"labels",       .rwBuf uintTy,          Semantic.none, some 2, some 0, .qIn⟩ ]
  , functions := [{
      attrs  := [.shaderCompute, .numthreads 256 1 1]
      name   := "main"
      params := [⟨"tid", .vec .uint 3, .svDispatchThreadId, none, none, .qIn⟩]
      body   :=
        [ .declInit uintTy "f" (.member (.var "tid") "x")
        , .ifNoElse (.bin ">=" (.var "f") (.member (.var "params") "num_faces"))
            [ .ret none ]
        , .declInit uintTy "base" (.bin "*" (.var "f") (.litUint 3))
        , .declInit uintTy "n"
            (.bin "+"
              (.bin "+" (.index (.var "edge_crossed") (.var "base"))
                        (.index (.var "edge_crossed")
                          (.bin "+" (.var "base") (.litUint 1))))
              (.index (.var "edge_crossed")
                (.bin "+" (.var "base") (.litUint 2))))
        , .assign (.index (.var "labels") (.var "f")) (.var "n")
        , .ret none ] }] }

def expected : String :=
"struct CutAlgoParams {
  uint num_faces;
};

[[vk::binding(0, 0)]]
ConstantBuffer<CutAlgoParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<uint> edge_crossed;
[[vk::binding(2, 0)]]
RWStructuredBuffer<uint> labels;

[shader(\"compute\")] [numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint f = tid.x;
  if ((f >= params.num_faces)) {
    return;
  }
  uint base = (f * 3u);
  uint n = ((edge_crossed[base] + edge_crossed[(base + 1u)]) + edge_crossed[(base + 2u)]);
  labels[f] = n;
  return;
}"

example : LeanSlang.emit shader = expected := by native_decide

example : shader.entryPointName = "main" := by native_decide

end Curvenet.SlangCodegen.CutAlgorithm
