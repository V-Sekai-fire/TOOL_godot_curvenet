import LeanSlang
import Curvenet.SlangCodegen.Common

/-!
# `Curvenet.SlangCodegen.CutMesh` — per-face curve-crossing predicate

Slang re-port of the per-face predicate at the heart of
`src/curvenet/cut_mesh.h`: does any cut curve cross this triangle?

For each triangle `t` and each cut segment `s`, the host bakes the
signed-distance-to-segment-plane for each of `t`'s three vertices.
A face crosses segment `s` when the three signed distances are not
all the same sign — i.e. the segment plane separates at least one
vertex from the other two. The kernel writes a flag per (face,
segment) pair indicating whether that segment cuts that face.

Bindings:
  0 — `ConstantBuffer<CutMeshParams> { uint num_faces; uint num_segments; }`
  1 — `StructuredBuffer<float> signed_dists` (length 3 * num_faces * num_segments)
  2 — `RWStructuredBuffer<uint> crosses`     (length num_faces * num_segments, 0/1)
-/

namespace Curvenet.SlangCodegen.CutMesh

open LeanSlang
open Curvenet.SlangCodegen.Common


def shader : SlangShaderModule :=
  { structs :=
      [ { name := "CutMeshParams"
        , fields :=
            [ ⟨"num_faces",    .scalar .uint, Semantic.none, none, none, .qIn⟩
            , ⟨"num_segments", .scalar .uint, Semantic.none, none, none, .qIn⟩ ] } ]
  , globals :=
      [ ⟨"params",       .const "CutMeshParams", Semantic.none, some 0, some 0, .qIn⟩
      , ⟨"signed_dists", .roBuf floatTy,         Semantic.none, some 1, some 0, .qIn⟩
      , ⟨"crosses",      .rwBuf uintTy,          Semantic.none, some 2, some 0, .qIn⟩ ]
  , functions := [{
      attrs  := [.shaderCompute, .numthreads 64 1 1]
      name   := "main"
      params := [⟨"tid", .vec .uint 3, .svDispatchThreadId, none, none, .qIn⟩]
      body   :=
        [ .declInit uintTy "gid" (.member (.var "tid") "x")
        , .declInit uintTy "total"
            (.bin "*" (.member (.var "params") "num_faces")
                      (.member (.var "params") "num_segments"))
        , .ifNoElse (.bin ">=" (.var "gid") (.var "total"))
            [ .ret none ]
        , .declInit uintTy "base" (.bin "*" (.var "gid") (.litUint 3))
        , .declInit floatTy "d0" (.index (.var "signed_dists") (.var "base"))
        , .declInit floatTy "d1"
            (.index (.var "signed_dists") (.bin "+" (.var "base") (.litUint 1)))
        , .declInit floatTy "d2"
            (.index (.var "signed_dists") (.bin "+" (.var "base") (.litUint 2)))
        , -- "crosses" iff signs are mixed
          .declInit uintTy "all_pos"
            (.ternary
              (.bin "&&"
                (.bin "&&" (.bin ">=" (.var "d0") (.litFloat 0.0))
                            (.bin ">=" (.var "d1") (.litFloat 0.0)))
                (.bin ">=" (.var "d2") (.litFloat 0.0)))
              (.litUint 1) (.litUint 0))
        , .declInit uintTy "all_neg"
            (.ternary
              (.bin "&&"
                (.bin "&&" (.bin "<=" (.var "d0") (.litFloat 0.0))
                            (.bin "<=" (.var "d1") (.litFloat 0.0)))
                (.bin "<=" (.var "d2") (.litFloat 0.0)))
              (.litUint 1) (.litUint 0))
        , .assign (.index (.var "crosses") (.var "gid"))
            (.ternary
              (.bin "||" (.bin "==" (.var "all_pos") (.litUint 1))
                          (.bin "==" (.var "all_neg") (.litUint 1)))
              (.litUint 0) (.litUint 1))
        , .ret none ] }] }

def expected : String :=
"struct CutMeshParams {
  uint num_faces;
  uint num_segments;
};

[[vk::binding(0, 0)]]
ConstantBuffer<CutMeshParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<float> signed_dists;
[[vk::binding(2, 0)]]
RWStructuredBuffer<uint> crosses;

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint gid = tid.x;
  uint total = (params.num_faces * params.num_segments);
  if ((gid >= total)) {
    return;
  }
  uint base = (gid * 3u);
  float d0 = signed_dists[base];
  float d1 = signed_dists[(base + 1u)];
  float d2 = signed_dists[(base + 2u)];
  uint all_pos = ((((d0 >= 0.000000) && (d1 >= 0.000000)) && (d2 >= 0.000000)) ? 1u : 0u);
  uint all_neg = ((((d0 <= 0.000000) && (d1 <= 0.000000)) && (d2 <= 0.000000)) ? 1u : 0u);
  crosses[gid] = (((all_pos == 1u) || (all_neg == 1u)) ? 0u : 1u);
  return;
}"

example : LeanSlang.emit shader = expected := by native_decide

example : shader.entryPointName = "main" := by native_decide

end Curvenet.SlangCodegen.CutMesh
