import LeanSlang

/-!
# `Curvenet.SlangCodegen.CurvenetBuilder` — per-segment frame assembly

Slang re-port of `src/curvenet/curvenet_builder.h`'s segment build
phase: for each curvenet segment, assemble the §3 scaled frame
`B · S` from the curve tangent + per-side normal/width data and
store it as a 3×3 matrix.

Frame columns: tangent (binormal axis), normal (curve normal),
binormal (cross). Scaled by per-side widths to encode the
half-width across the curve.

Bindings:
  0 — `ConstantBuffer<CnetBuilderParams> { uint num_segments; }`
  1 — `StructuredBuffer<float3> tangents`  (length num_segments)
  2 — `StructuredBuffer<float3> normals`   (length num_segments)
  3 — `StructuredBuffer<float>  widths`    (length num_segments)
  4 — `RWStructuredBuffer<float> frames`   (9 floats per segment)
-/

namespace Curvenet.SlangCodegen.CurvenetBuilder

open LeanSlang

private def floatTy : SlangType := .scalar .float
private def f3Ty    : SlangType := .vec .float 3
private def m3Ty    : SlangType := .mat .float 3 3
private def uintTy  : SlangType := .scalar .uint

private def storeM3 (base : SlangExpr) (m : SlangExpr) : List SlangStmt :=
  [ .assign (.index (.var "frames") (.bin "+" base (.litUint 0))) (.member m "_m00")
  , .assign (.index (.var "frames") (.bin "+" base (.litUint 1))) (.member m "_m01")
  , .assign (.index (.var "frames") (.bin "+" base (.litUint 2))) (.member m "_m02")
  , .assign (.index (.var "frames") (.bin "+" base (.litUint 3))) (.member m "_m10")
  , .assign (.index (.var "frames") (.bin "+" base (.litUint 4))) (.member m "_m11")
  , .assign (.index (.var "frames") (.bin "+" base (.litUint 5))) (.member m "_m12")
  , .assign (.index (.var "frames") (.bin "+" base (.litUint 6))) (.member m "_m20")
  , .assign (.index (.var "frames") (.bin "+" base (.litUint 7))) (.member m "_m21")
  , .assign (.index (.var "frames") (.bin "+" base (.litUint 8))) (.member m "_m22") ]

def shader : SlangShaderModule :=
  { structs :=
      [ { name := "CnetBuilderParams"
        , fields := [⟨"num_segments", .scalar .uint, Semantic.none, none, none, .qIn⟩] } ]
  , globals :=
      [ ⟨"params",   .const "CnetBuilderParams", Semantic.none, some 0, some 0, .qIn⟩
      , ⟨"tangents", .roBuf f3Ty,                Semantic.none, some 1, some 0, .qIn⟩
      , ⟨"normals",  .roBuf f3Ty,                Semantic.none, some 2, some 0, .qIn⟩
      , ⟨"widths",   .roBuf floatTy,             Semantic.none, some 3, some 0, .qIn⟩
      , ⟨"frames",   .rwBuf floatTy,             Semantic.none, some 4, some 0, .qIn⟩ ]
  , functions := [{
      attrs  := [.shaderCompute, .numthreads 64 1 1]
      name   := "main"
      params := [⟨"tid", .vec .uint 3, .svDispatchThreadId, none, none, .qIn⟩]
      body   :=
        [ .declInit uintTy "i" (.member (.var "tid") "x")
        , .ifNoElse (.bin ">=" (.var "i") (.member (.var "params") "num_segments"))
            [ .ret none ]
        , .declInit f3Ty "t" (.call "normalize" [.index (.var "tangents") (.var "i")])
        , .declInit f3Ty "n" (.call "normalize" [.index (.var "normals") (.var "i")])
        , .declInit f3Ty "b" (.call "cross" [.var "t", .var "n"])
        , .declInit floatTy "w" (.index (.var "widths") (.var "i"))
        , -- Scale axes by half-width — encodes the §3 frame "S" diagonal.
          .declInit f3Ty "ts" (.bin "*" (.var "w") (.var "t"))
        , .declInit f3Ty "ns" (.bin "*" (.var "w") (.var "n"))
        , .declInit f3Ty "bs" (.bin "*" (.var "w") (.var "b"))
        , -- Frame columns t, n, b → row-major float3x3.
          .declInit m3Ty "F"
            (.call "float3x3"
              [ .member (.var "ts") "x", .member (.var "ns") "x", .member (.var "bs") "x"
              , .member (.var "ts") "y", .member (.var "ns") "y", .member (.var "bs") "y"
              , .member (.var "ts") "z", .member (.var "ns") "z", .member (.var "bs") "z" ])
        , .declInit uintTy "base" (.bin "*" (.var "i") (.litUint 9))
        ] ++ storeM3 (.var "base") (.var "F") ++
        [ .ret none ] }] }

def expected : String :=
"struct CnetBuilderParams {
  uint num_segments;
};

[[vk::binding(0, 0)]]
ConstantBuffer<CnetBuilderParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<float3> tangents;
[[vk::binding(2, 0)]]
StructuredBuffer<float3> normals;
[[vk::binding(3, 0)]]
StructuredBuffer<float> widths;
[[vk::binding(4, 0)]]
RWStructuredBuffer<float> frames;

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint i = tid.x;
  if ((i >= params.num_segments)) {
    return;
  }
  float3 t = normalize(tangents[i]);
  float3 n = normalize(normals[i]);
  float3 b = cross(t, n);
  float w = widths[i];
  float3 ts = (w * t);
  float3 ns = (w * n);
  float3 bs = (w * b);
  float3x3 F = float3x3(ts.x, ns.x, bs.x, ts.y, ns.y, bs.y, ts.z, ns.z, bs.z);
  uint base = (i * 9u);
  frames[(base + 0u)] = F._m00;
  frames[(base + 1u)] = F._m01;
  frames[(base + 2u)] = F._m02;
  frames[(base + 3u)] = F._m10;
  frames[(base + 4u)] = F._m11;
  frames[(base + 5u)] = F._m12;
  frames[(base + 6u)] = F._m20;
  frames[(base + 7u)] = F._m21;
  frames[(base + 8u)] = F._m22;
  return;
}"

example : LeanSlang.emit shader = expected := by native_decide

example : shader.entryPointName = "main" := by native_decide

end Curvenet.SlangCodegen.CurvenetBuilder
