import LeanSlang

/-!
# `Curvenet.SlangCodegen.DeformSolve` — per-vertex Poisson RHS assembly

Slang re-port of the §4.3 Stage 2 RHS-assembly step from
`src/curvenet/deform_solve.h`. Stage 2 solves
  L · X_v = div(F)
where `F` is the per-segment deformation gradient already computed
by `ScaledFrames` / `SegmentGradient` / `IntersectionFrames`. This
kernel computes `div(F)[i]` per vertex by gathering F contributions
across the vertex's incident face segments.

Per-thread work (one per vertex):
  div_x[i] = Σ over incident_segs s   F[s] · (rest_q[s] - rest_p[s])

The actual Poisson solve afterwards uses the same SpMV + CG chain
as HarmonicSolve.

Bindings (set 0):
  0 — `ConstantBuffer<DeformParams> { uint num_verts; }`
  1 — `StructuredBuffer<int>   incident_start` (CSR row ptr, length n+1)
  2 — `StructuredBuffer<int>   incident_idx`   (segment indices, length nnz)
  3 — `StructuredBuffer<float> F_seg` (9 floats per segment)
  4 — `StructuredBuffer<float3> rest_p`   (segment start)
  5 — `StructuredBuffer<float3> rest_q`   (segment end)
  6 — `RWStructuredBuffer<float3> div_F` (length num_verts, output)
-/

namespace Curvenet.SlangCodegen.DeformSolve

open LeanSlang

private def floatTy : SlangType := .scalar .float
private def f3Ty    : SlangType := .vec .float 3
private def m3Ty    : SlangType := .mat .float 3 3
private def intTy   : SlangType := .scalar .int
private def uintTy  : SlangType := .scalar .uint

private def loadM3 : SlangFunctionDecl :=
  { attrs := []
  , retType := m3Ty
  , name := "loadM3"
  , params :=
      [ ⟨"buf",  .roBuf floatTy, Semantic.none, none, none, .qIn⟩
      , ⟨"base", uintTy,         Semantic.none, none, none, .qIn⟩ ]
  , body :=
      [ .retExpr (.call "float3x3"
          [ .index (.var "buf") (.bin "+" (.var "base") (.litUint 0))
          , .index (.var "buf") (.bin "+" (.var "base") (.litUint 1))
          , .index (.var "buf") (.bin "+" (.var "base") (.litUint 2))
          , .index (.var "buf") (.bin "+" (.var "base") (.litUint 3))
          , .index (.var "buf") (.bin "+" (.var "base") (.litUint 4))
          , .index (.var "buf") (.bin "+" (.var "base") (.litUint 5))
          , .index (.var "buf") (.bin "+" (.var "base") (.litUint 6))
          , .index (.var "buf") (.bin "+" (.var "base") (.litUint 7))
          , .index (.var "buf") (.bin "+" (.var "base") (.litUint 8)) ]) ] }

private def mainEntry : SlangFunctionDecl :=
  { attrs  := [.shaderCompute, .numthreads 256 1 1]
  , name   := "main"
  , params := [⟨"tid", .vec .uint 3, .svDispatchThreadId, none, none, .qIn⟩]
  , body   :=
      [ .declInit uintTy "v" (.member (.var "tid") "x")
      , .ifNoElse (.bin ">=" (.var "v") (.member (.var "params") "num_verts"))
          [ .ret none ]
      , .declInit uintTy "rs"
          (.call "uint" [.index (.var "incident_start") (.var "v")])
      , .declInit uintTy "re"
          (.call "uint" [.index (.var "incident_start")
            (.bin "+" (.var "v") (.litUint 1))])
      , .declInit f3Ty "acc"
          (.call "float3" [.litFloat 0.0, .litFloat 0.0, .litFloat 0.0])
      , .forCount "p" (.var "rs") (.var "re")
          [ .declInit uintTy "s"
              (.call "uint" [.index (.var "incident_idx") (.var "p")])
          , .declInit f3Ty "edge"
              (.bin "-" (.index (.var "rest_q") (.var "s"))
                        (.index (.var "rest_p") (.var "s")))
          , .declInit uintTy "fbase" (.bin "*" (.var "s") (.litUint 9))
          , .declInit m3Ty "F" (.call "loadM3" [.var "F_seg", .var "fbase"])
          , .assign (.var "acc")
              (.bin "+" (.var "acc")
                (.call "mul" [.var "F", .var "edge"])) ]
      , .assign (.index (.var "div_F") (.var "v")) (.var "acc")
      , .ret none ] }

def shader : SlangShaderModule :=
  { structs :=
      [ { name := "DeformParams"
        , fields := [⟨"num_verts", .scalar .uint, Semantic.none, none, none, .qIn⟩] } ]
  , globals :=
      [ ⟨"params",         .const "DeformParams", Semantic.none, some 0, some 0, .qIn⟩
      , ⟨"incident_start", .roBuf intTy,          Semantic.none, some 1, some 0, .qIn⟩
      , ⟨"incident_idx",   .roBuf intTy,          Semantic.none, some 2, some 0, .qIn⟩
      , ⟨"F_seg",          .roBuf floatTy,        Semantic.none, some 3, some 0, .qIn⟩
      , ⟨"rest_p",         .roBuf f3Ty,           Semantic.none, some 4, some 0, .qIn⟩
      , ⟨"rest_q",         .roBuf f3Ty,           Semantic.none, some 5, some 0, .qIn⟩
      , ⟨"div_F",          .rwBuf f3Ty,           Semantic.none, some 6, some 0, .qIn⟩ ]
  , functions := [loadM3, mainEntry] }

def expected : String :=
"struct DeformParams {
  uint num_verts;
};

[[vk::binding(0, 0)]]
ConstantBuffer<DeformParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<int> incident_start;
[[vk::binding(2, 0)]]
StructuredBuffer<int> incident_idx;
[[vk::binding(3, 0)]]
StructuredBuffer<float> F_seg;
[[vk::binding(4, 0)]]
StructuredBuffer<float3> rest_p;
[[vk::binding(5, 0)]]
StructuredBuffer<float3> rest_q;
[[vk::binding(6, 0)]]
RWStructuredBuffer<float3> div_F;

float3x3 loadM3(StructuredBuffer<float> buf, uint base) {
  return float3x3(buf[(base + 0u)], buf[(base + 1u)], buf[(base + 2u)], buf[(base + 3u)], buf[(base + 4u)], buf[(base + 5u)], buf[(base + 6u)], buf[(base + 7u)], buf[(base + 8u)]);
}

[shader(\"compute\")] [numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint v = tid.x;
  if ((v >= params.num_verts)) {
    return;
  }
  uint rs = uint(incident_start[v]);
  uint re = uint(incident_start[(v + 1u)]);
  float3 acc = float3(0.000000, 0.000000, 0.000000);
  for (uint p = rs; p < re; ++p) {
    uint s = uint(incident_idx[p]);
    float3 edge = (rest_q[s] - rest_p[s]);
    uint fbase = (s * 9u);
    float3x3 F = loadM3(F_seg, fbase);
    acc = (acc + mul(F, edge));
  }
  div_F[v] = acc;
  return;
}"

example : LeanSlang.emit shader = expected := by native_decide

example : shader.entryPointName = "main" := by native_decide

end Curvenet.SlangCodegen.DeformSolve
