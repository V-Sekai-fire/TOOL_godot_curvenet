import LeanSlang

/-!
# `Curvenet.SlangCodegen.SegmentGradient` — isolated-segment F dispatcher

Slang re-port of `src/curvenet/segment_gradient.h`. The
`isolated(rest_p, rest_q, posed_p, posed_q)` formula computes a
3×3 segment deformation gradient `F = (l_posed / l_rest) ·
R(t̂_rest → t̂_posed)` where R is the smallest rotation aligning the
unit tangents. Per-segment, one thread.

The `intersection_pair` form (paired plus/minus frames) is just
two independent calls to ScaledFrames.deformation_gradient on the
host; no kernel needed beyond what ScaledFrames already provides.

Bindings (set 0):
  0 — `ConstantBuffer<SegGradParams> { uint num_segments; }`
  1 — `StructuredBuffer<float3> rest_p`   (segment start, rest pose)
  2 — `StructuredBuffer<float3> rest_q`   (segment end,   rest pose)
  3 — `StructuredBuffer<float3> posed_p`  (segment start, posed)
  4 — `StructuredBuffer<float3> posed_q`  (segment end,   posed)
  5 — `RWStructuredBuffer<float> F`       (9 floats per segment)
-/

namespace Curvenet.SlangCodegen.SegmentGradient

open LeanSlang

private def floatTy : SlangType := .scalar .float
private def f3Ty    : SlangType := .vec .float 3
private def m3Ty    : SlangType := .mat .float 3 3
private def uintTy  : SlangType := .scalar .uint

/-- `skew(v) = [ 0 -vz vy; vz 0 -vx; -vy vx 0 ]`. -/
private def skew3 : SlangFunctionDecl :=
  { attrs := []
  , retType := m3Ty
  , name := "skew3"
  , params := [⟨"v", f3Ty, Semantic.none, none, none, .qIn⟩]
  , body :=
      [ .retExpr (.call "float3x3"
          [ .litFloat 0.0
          , .un "-" (.member (.var "v") "z")
          , .member (.var "v") "y"
          , .member (.var "v") "z"
          , .litFloat 0.0
          , .un "-" (.member (.var "v") "x")
          , .un "-" (.member (.var "v") "y")
          , .member (.var "v") "x"
          , .litFloat 0.0 ]) ] }

/-- `smallest_rotation(from, to) = I + K + K² / (1 + cos)` where
    K = skew(from × to). Antiparallel inputs collapse to zero. -/
private def smallest_rotation : SlangFunctionDecl :=
  { attrs := []
  , retType := m3Ty
  , name := "smallest_rotation"
  , params :=
      [ ⟨"fromV", f3Ty, Semantic.none, none, none, .qIn⟩
      , ⟨"toV",   f3Ty, Semantic.none, none, none, .qIn⟩ ]
  , body :=
      [ .declInit f3Ty "cv" (.call "cross" [.var "fromV", .var "toV"])
      , .declInit floatTy "ct" (.call "dot" [.var "fromV", .var "toV"])
      , .declInit m3Ty "K"  (.call "skew3" [.var "cv"])
      , .declInit m3Ty "K2" (.call "mul"   [.var "K", .var "K"])
      , .declInit floatTy "denom" (.bin "+" (.litFloat 1.0) (.var "ct"))
      , .declInit floatTy "invD"
          (.ternary (.bin "<" (.call "abs" [.var "denom"]) (.litFloat 1e-30))
            (.litFloat 0.0)
            (.bin "/" (.litFloat 1.0) (.var "denom")))
      , .retExpr
          (.bin "+"
            (.bin "+"
              (.call "float3x3"
                [.litFloat 1.0, .litFloat 0.0, .litFloat 0.0
                ,.litFloat 0.0, .litFloat 1.0, .litFloat 0.0
                ,.litFloat 0.0, .litFloat 0.0, .litFloat 1.0])
              (.var "K"))
            (.bin "*" (.var "invD") (.var "K2"))) ] }

private def storeM3 (base : SlangExpr) (m : SlangExpr) : List SlangStmt :=
  [ .assign (.index (.var "F") (.bin "+" base (.litUint 0))) (.member m "_m00")
  , .assign (.index (.var "F") (.bin "+" base (.litUint 1))) (.member m "_m01")
  , .assign (.index (.var "F") (.bin "+" base (.litUint 2))) (.member m "_m02")
  , .assign (.index (.var "F") (.bin "+" base (.litUint 3))) (.member m "_m10")
  , .assign (.index (.var "F") (.bin "+" base (.litUint 4))) (.member m "_m11")
  , .assign (.index (.var "F") (.bin "+" base (.litUint 5))) (.member m "_m12")
  , .assign (.index (.var "F") (.bin "+" base (.litUint 6))) (.member m "_m20")
  , .assign (.index (.var "F") (.bin "+" base (.litUint 7))) (.member m "_m21")
  , .assign (.index (.var "F") (.bin "+" base (.litUint 8))) (.member m "_m22") ]

private def mainEntry : SlangFunctionDecl :=
  { attrs  := [.shaderCompute, .numthreads 64 1 1]
  , name   := "main"
  , params := [⟨"tid", .vec .uint 3, .svDispatchThreadId, none, none, .qIn⟩]
  , body   :=
      [ .declInit uintTy "t" (.member (.var "tid") "x")
      , .ifNoElse (.bin ">=" (.var "t") (.member (.var "params") "num_segments"))
          [ .ret none ]
      , .declInit f3Ty "rp" (.index (.var "rest_p") (.var "t"))
      , .declInit f3Ty "rq" (.index (.var "rest_q") (.var "t"))
      , .declInit f3Ty "pp" (.index (.var "posed_p") (.var "t"))
      , .declInit f3Ty "pq" (.index (.var "posed_q") (.var "t"))
      , .declInit f3Ty "rd" (.bin "-" (.var "rq") (.var "rp"))
      , .declInit f3Ty "pd" (.bin "-" (.var "pq") (.var "pp"))
      , .declInit floatTy "rl" (.call "length" [.var "rd"])
      , .declInit floatTy "pl" (.call "length" [.var "pd"])
      , .declInit floatTy "ratio"
          (.ternary (.bin "<" (.var "rl") (.litFloat 1e-30))
            (.litFloat 0.0)
            (.bin "/" (.var "pl") (.var "rl")))
      , .declInit f3Ty "rt"
          (.ternary (.bin "<" (.var "rl") (.litFloat 1e-30))
            (.call "float3" [.litFloat 0.0, .litFloat 0.0, .litFloat 0.0])
            (.bin "/" (.var "rd") (.var "rl")))
      , .declInit f3Ty "pt"
          (.ternary (.bin "<" (.var "pl") (.litFloat 1e-30))
            (.call "float3" [.litFloat 0.0, .litFloat 0.0, .litFloat 0.0])
            (.bin "/" (.var "pd") (.var "pl")))
      , .declInit m3Ty "R"
          (.call "smallest_rotation" [.var "rt", .var "pt"])
      , .declInit m3Ty "Fmat" (.bin "*" (.var "ratio") (.var "R"))
      , .declInit uintTy "base" (.bin "*" (.var "t") (.litUint 9))
      ] ++ storeM3 (.var "base") (.var "Fmat") ++
      [ .ret none ] }

def shader : SlangShaderModule :=
  { structs :=
      [ { name := "SegGradParams"
        , fields := [⟨"num_segments", .scalar .uint, Semantic.none, none, none, .qIn⟩] } ]
  , globals :=
      [ ⟨"params",  .const "SegGradParams", Semantic.none, some 0, some 0, .qIn⟩
      , ⟨"rest_p",  .roBuf f3Ty,            Semantic.none, some 1, some 0, .qIn⟩
      , ⟨"rest_q",  .roBuf f3Ty,            Semantic.none, some 2, some 0, .qIn⟩
      , ⟨"posed_p", .roBuf f3Ty,            Semantic.none, some 3, some 0, .qIn⟩
      , ⟨"posed_q", .roBuf f3Ty,            Semantic.none, some 4, some 0, .qIn⟩
      , ⟨"F",       .rwBuf floatTy,         Semantic.none, some 5, some 0, .qIn⟩ ]
  , functions := [skew3, smallest_rotation, mainEntry] }

def expected : String :=
"struct SegGradParams {
  uint num_segments;
};

[[vk::binding(0, 0)]]
ConstantBuffer<SegGradParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<float3> rest_p;
[[vk::binding(2, 0)]]
StructuredBuffer<float3> rest_q;
[[vk::binding(3, 0)]]
StructuredBuffer<float3> posed_p;
[[vk::binding(4, 0)]]
StructuredBuffer<float3> posed_q;
[[vk::binding(5, 0)]]
RWStructuredBuffer<float> F;

float3x3 skew3(float3 v) {
  return float3x3(0.000000, (-v.z), v.y, v.z, 0.000000, (-v.x), (-v.y), v.x, 0.000000);
}

float3x3 smallest_rotation(float3 fromV, float3 toV) {
  float3 cv = cross(fromV, toV);
  float ct = dot(fromV, toV);
  float3x3 K = skew3(cv);
  float3x3 K2 = mul(K, K);
  float denom = (1.000000 + ct);
  float invD = ((abs(denom) < 0.000000) ? 0.000000 : (1.000000 / denom));
  return ((float3x3(1.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 1.000000) + K) + (invD * K2));
}

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint t = tid.x;
  if ((t >= params.num_segments)) {
    return;
  }
  float3 rp = rest_p[t];
  float3 rq = rest_q[t];
  float3 pp = posed_p[t];
  float3 pq = posed_q[t];
  float3 rd = (rq - rp);
  float3 pd = (pq - pp);
  float rl = length(rd);
  float pl = length(pd);
  float ratio = ((rl < 0.000000) ? 0.000000 : (pl / rl));
  float3 rt = ((rl < 0.000000) ? float3(0.000000, 0.000000, 0.000000) : (rd / rl));
  float3 pt = ((pl < 0.000000) ? float3(0.000000, 0.000000, 0.000000) : (pd / pl));
  float3x3 R = smallest_rotation(rt, pt);
  float3x3 Fmat = (ratio * R);
  uint base = (t * 9u);
  F[(base + 0u)] = Fmat._m00;
  F[(base + 1u)] = Fmat._m01;
  F[(base + 2u)] = Fmat._m02;
  F[(base + 3u)] = Fmat._m10;
  F[(base + 4u)] = Fmat._m11;
  F[(base + 5u)] = Fmat._m12;
  F[(base + 6u)] = Fmat._m20;
  F[(base + 7u)] = Fmat._m21;
  F[(base + 8u)] = Fmat._m22;
  return;
}"

example : LeanSlang.emit shader = expected := by native_decide

example : shader.entryPointName = "main" := by native_decide

end Curvenet.SlangCodegen.SegmentGradient
