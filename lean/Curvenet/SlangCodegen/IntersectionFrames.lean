import LeanSlang

/-!
# `Curvenet.SlangCodegen.IntersectionFrames` — paired plus/minus F kernels

Slang re-port of `src/curvenet/intersection_frames.h` /
`segment_gradient::intersection_pair`. For each intersection,
two `F = posed · rest⁻¹` matrices are produced — one for the plus
side, one for the minus side. The kernel is two simultaneous calls
to ScaledFrames-style 3×3 inverse + multiply, packed into a single
thread per intersection.

Bindings (set 0):
  0 — `ConstantBuffer<IsectFramesParams> { uint num_isects; }`
  1 — `StructuredBuffer<float> rest_plus`    (9 per isect)
  2 — `StructuredBuffer<float> posed_plus`   (9 per isect)
  3 — `StructuredBuffer<float> rest_minus`   (9 per isect)
  4 — `StructuredBuffer<float> posed_minus`  (9 per isect)
  5 — `RWStructuredBuffer<float> F_plus`     (9 per isect, output)
  6 — `RWStructuredBuffer<float> F_minus`    (9 per isect, output)
-/

namespace Curvenet.SlangCodegen.IntersectionFrames

open LeanSlang

private def floatTy : SlangType := .scalar .float
private def m3Ty    : SlangType := .mat .float 3 3
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

/-- Inline 3×3 inverse with cofactor + zero-determinant guard. -/
private def inv3x3 : SlangFunctionDecl :=
  { attrs := []
  , retType := m3Ty
  , name := "inv3x3"
  , params := [⟨"m", m3Ty, Semantic.none, none, none, .qIn⟩]
  , body :=
      [ .declInit floatTy "a" (.member (.var "m") "_m00")
      , .declInit floatTy "b" (.member (.var "m") "_m01")
      , .declInit floatTy "c" (.member (.var "m") "_m02")
      , .declInit floatTy "d" (.member (.var "m") "_m10")
      , .declInit floatTy "e" (.member (.var "m") "_m11")
      , .declInit floatTy "f" (.member (.var "m") "_m12")
      , .declInit floatTy "g" (.member (.var "m") "_m20")
      , .declInit floatTy "h" (.member (.var "m") "_m21")
      , .declInit floatTy "i" (.member (.var "m") "_m22")
      , .declInit floatTy "det"
          (.bin "+"
            (.bin "-"
              (.bin "*" (.var "a")
                (.bin "-" (.bin "*" (.var "e") (.var "i"))
                          (.bin "*" (.var "f") (.var "h"))))
              (.bin "*" (.var "b")
                (.bin "-" (.bin "*" (.var "d") (.var "i"))
                          (.bin "*" (.var "f") (.var "g")))))
            (.bin "*" (.var "c")
              (.bin "-" (.bin "*" (.var "d") (.var "h"))
                        (.bin "*" (.var "e") (.var "g")))))
      , .declInit floatTy "invDet"
          (.ternary (.bin "<" (.call "abs" [.var "det"]) (.litFloat 1e-30))
            (.litFloat 0.0)
            (.bin "/" (.litFloat 1.0) (.var "det")))
      , .retExpr (.call "float3x3"
          [ .bin "*" (.var "invDet") (.bin "-" (.bin "*" (.var "e") (.var "i")) (.bin "*" (.var "f") (.var "h")))
          , .bin "*" (.var "invDet") (.bin "-" (.bin "*" (.var "c") (.var "h")) (.bin "*" (.var "b") (.var "i")))
          , .bin "*" (.var "invDet") (.bin "-" (.bin "*" (.var "b") (.var "f")) (.bin "*" (.var "c") (.var "e")))
          , .bin "*" (.var "invDet") (.bin "-" (.bin "*" (.var "f") (.var "g")) (.bin "*" (.var "d") (.var "i")))
          , .bin "*" (.var "invDet") (.bin "-" (.bin "*" (.var "a") (.var "i")) (.bin "*" (.var "c") (.var "g")))
          , .bin "*" (.var "invDet") (.bin "-" (.bin "*" (.var "c") (.var "d")) (.bin "*" (.var "a") (.var "f")))
          , .bin "*" (.var "invDet") (.bin "-" (.bin "*" (.var "d") (.var "h")) (.bin "*" (.var "e") (.var "g")))
          , .bin "*" (.var "invDet") (.bin "-" (.bin "*" (.var "b") (.var "g")) (.bin "*" (.var "a") (.var "h")))
          , .bin "*" (.var "invDet") (.bin "-" (.bin "*" (.var "a") (.var "e")) (.bin "*" (.var "b") (.var "d"))) ]) ] }

private def storeM3 (target : String) (base : SlangExpr) (m : SlangExpr) : List SlangStmt :=
  [ .assign (.index (.var target) (.bin "+" base (.litUint 0))) (.member m "_m00")
  , .assign (.index (.var target) (.bin "+" base (.litUint 1))) (.member m "_m01")
  , .assign (.index (.var target) (.bin "+" base (.litUint 2))) (.member m "_m02")
  , .assign (.index (.var target) (.bin "+" base (.litUint 3))) (.member m "_m10")
  , .assign (.index (.var target) (.bin "+" base (.litUint 4))) (.member m "_m11")
  , .assign (.index (.var target) (.bin "+" base (.litUint 5))) (.member m "_m12")
  , .assign (.index (.var target) (.bin "+" base (.litUint 6))) (.member m "_m20")
  , .assign (.index (.var target) (.bin "+" base (.litUint 7))) (.member m "_m21")
  , .assign (.index (.var target) (.bin "+" base (.litUint 8))) (.member m "_m22") ]

private def mainEntry : SlangFunctionDecl :=
  { attrs  := [.shaderCompute, .numthreads 64 1 1]
  , name   := "main"
  , params := [⟨"tid", .vec .uint 3, .svDispatchThreadId, none, none, .qIn⟩]
  , body   :=
      [ .declInit uintTy "t" (.member (.var "tid") "x")
      , .ifNoElse (.bin ">=" (.var "t") (.member (.var "params") "num_isects"))
          [ .ret none ]
      , .declInit uintTy "base" (.bin "*" (.var "t") (.litUint 9))
      , .declInit m3Ty "rp" (.call "loadM3" [.var "rest_plus",   .var "base"])
      , .declInit m3Ty "pp" (.call "loadM3" [.var "posed_plus",  .var "base"])
      , .declInit m3Ty "rm" (.call "loadM3" [.var "rest_minus",  .var "base"])
      , .declInit m3Ty "pm" (.call "loadM3" [.var "posed_minus", .var "base"])
      , .declInit m3Ty "Fp"
          (.call "mul" [.var "pp", .call "inv3x3" [.var "rp"]])
      , .declInit m3Ty "Fm"
          (.call "mul" [.var "pm", .call "inv3x3" [.var "rm"]])
      ] ++ storeM3 "F_plus"  (.var "base") (.var "Fp")
        ++ storeM3 "F_minus" (.var "base") (.var "Fm")
        ++ [ .ret none ] }

def shader : SlangShaderModule :=
  { structs :=
      [ { name := "IsectFramesParams"
        , fields := [⟨"num_isects", .scalar .uint, Semantic.none, none, none, .qIn⟩] } ]
  , globals :=
      [ ⟨"params",       .const "IsectFramesParams", Semantic.none, some 0, some 0, .qIn⟩
      , ⟨"rest_plus",    .roBuf floatTy,             Semantic.none, some 1, some 0, .qIn⟩
      , ⟨"posed_plus",   .roBuf floatTy,             Semantic.none, some 2, some 0, .qIn⟩
      , ⟨"rest_minus",   .roBuf floatTy,             Semantic.none, some 3, some 0, .qIn⟩
      , ⟨"posed_minus",  .roBuf floatTy,             Semantic.none, some 4, some 0, .qIn⟩
      , ⟨"F_plus",       .rwBuf floatTy,             Semantic.none, some 5, some 0, .qIn⟩
      , ⟨"F_minus",      .rwBuf floatTy,             Semantic.none, some 6, some 0, .qIn⟩ ]
  , functions := [loadM3, inv3x3, mainEntry] }

def expected : String :=
"struct IsectFramesParams {
  uint num_isects;
};

[[vk::binding(0, 0)]]
ConstantBuffer<IsectFramesParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<float> rest_plus;
[[vk::binding(2, 0)]]
StructuredBuffer<float> posed_plus;
[[vk::binding(3, 0)]]
StructuredBuffer<float> rest_minus;
[[vk::binding(4, 0)]]
StructuredBuffer<float> posed_minus;
[[vk::binding(5, 0)]]
RWStructuredBuffer<float> F_plus;
[[vk::binding(6, 0)]]
RWStructuredBuffer<float> F_minus;

float3x3 loadM3(StructuredBuffer<float> buf, uint base) {
  return float3x3(buf[(base + 0u)], buf[(base + 1u)], buf[(base + 2u)], buf[(base + 3u)], buf[(base + 4u)], buf[(base + 5u)], buf[(base + 6u)], buf[(base + 7u)], buf[(base + 8u)]);
}

float3x3 inv3x3(float3x3 m) {
  float a = m._m00;
  float b = m._m01;
  float c = m._m02;
  float d = m._m10;
  float e = m._m11;
  float f = m._m12;
  float g = m._m20;
  float h = m._m21;
  float i = m._m22;
  float det = (((a * ((e * i) - (f * h))) - (b * ((d * i) - (f * g)))) + (c * ((d * h) - (e * g))));
  float invDet = ((abs(det) < 0.000000) ? 0.000000 : (1.000000 / det));
  return float3x3((invDet * ((e * i) - (f * h))), (invDet * ((c * h) - (b * i))), (invDet * ((b * f) - (c * e))), (invDet * ((f * g) - (d * i))), (invDet * ((a * i) - (c * g))), (invDet * ((c * d) - (a * f))), (invDet * ((d * h) - (e * g))), (invDet * ((b * g) - (a * h))), (invDet * ((a * e) - (b * d))));
}

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint t = tid.x;
  if ((t >= params.num_isects)) {
    return;
  }
  uint base = (t * 9u);
  float3x3 rp = loadM3(rest_plus, base);
  float3x3 pp = loadM3(posed_plus, base);
  float3x3 rm = loadM3(rest_minus, base);
  float3x3 pm = loadM3(posed_minus, base);
  float3x3 Fp = mul(pp, inv3x3(rp));
  float3x3 Fm = mul(pm, inv3x3(rm));
  F_plus[(base + 0u)] = Fp._m00;
  F_plus[(base + 1u)] = Fp._m01;
  F_plus[(base + 2u)] = Fp._m02;
  F_plus[(base + 3u)] = Fp._m10;
  F_plus[(base + 4u)] = Fp._m11;
  F_plus[(base + 5u)] = Fp._m12;
  F_plus[(base + 6u)] = Fp._m20;
  F_plus[(base + 7u)] = Fp._m21;
  F_plus[(base + 8u)] = Fp._m22;
  F_minus[(base + 0u)] = Fm._m00;
  F_minus[(base + 1u)] = Fm._m01;
  F_minus[(base + 2u)] = Fm._m02;
  F_minus[(base + 3u)] = Fm._m10;
  F_minus[(base + 4u)] = Fm._m11;
  F_minus[(base + 5u)] = Fm._m12;
  F_minus[(base + 6u)] = Fm._m20;
  F_minus[(base + 7u)] = Fm._m21;
  F_minus[(base + 8u)] = Fm._m22;
  return;
}"

example : LeanSlang.emit shader = expected := by native_decide

example : shader.entryPointName = "main" := by native_decide

end Curvenet.SlangCodegen.IntersectionFrames
