import LeanSlang
import Curvenet.SlangCodegen.Common

/-!
# `Curvenet.SlangCodegen.PolygonLaplacian` — per-triangle cot Laplacian

Slang re-port of `src/curvenet/polygon_laplacian.h` (the
`triangle_cot_laplacian` core; n-gon fan triangulation is left to
the host driver).

One thread per triangle. For each triangle (a, b, c) computes the
three cotangents at apexes a, b, c and writes the 3×3 cot Laplacian:

      [  w_ab+w_ac     -w_ab        -w_ac      ]
      [    -w_ab     w_ab+w_bc      -w_bc      ]
      [    -w_ac      -w_bc       w_ac+w_bc    ]

where `w_xy = 0.5 · cot_at_apex_opposite_to_xy`. The output matrix
buffer is 9 floats per triangle, row-major.

Bindings (set 0):
  0 — `ConstantBuffer<PolyLapParams> { uint num_triangles; }`
  1 — `StructuredBuffer<float3> positions`           (per-vertex)
  2 — `StructuredBuffer<int> tri_indices`            (3 ints per triangle, length = 3 * num_triangles)
  3 — `RWStructuredBuffer<float> matrices`           (9 floats per triangle)
-/

namespace Curvenet.SlangCodegen.PolygonLaplacian

open LeanSlang
open Curvenet.SlangCodegen.Common


/-- `cot_at_apex(apex, o1, o2) = dot(a, b) / length(cross(a, b))`
    where a = o1 - apex, b = o2 - apex. -/
private def cot_at_apex : SlangFunctionDecl :=
  { attrs   := []
  , retType := floatTy
  , name    := "cot_at_apex"
  , params  :=
      [ ⟨"apex", f3Ty, Semantic.none, none, none, .qIn⟩
      , ⟨"o1",   f3Ty, Semantic.none, none, none, .qIn⟩
      , ⟨"o2",   f3Ty, Semantic.none, none, none, .qIn⟩ ]
  , body    :=
      [ .declInit f3Ty "a" (.bin "-" (.var "o1") (.var "apex"))
      , .declInit f3Ty "b" (.bin "-" (.var "o2") (.var "apex"))
      , .declInit floatTy "d" (.call "dot" [.var "a", .var "b"])
      , .declInit floatTy "cl"
          (.call "length" [.call "cross" [.var "a", .var "b"]])
      , .retExpr (.bin "/" (.var "d") (.var "cl")) ] }

/-- Helper: write `m[i, j] = v` for the per-triangle 3×3 row-major
    matrix at base offset `base = 9 * tri`. -/
private def setM (base i j : SlangExpr) (v : SlangExpr) : SlangStmt :=
  .assign (.index (.var "matrices")
            (.bin "+" base (.bin "+" (.bin "*" i (.litUint 3)) j)))
          v

private def mainEntry : SlangFunctionDecl :=
  { attrs  := [.shaderCompute, .numthreads 64 1 1]
  , name   := "main"
  , params := [⟨"tid", .vec .uint 3, .svDispatchThreadId, none, none, .qIn⟩]
  , body   :=
      [ .declInit uintTy "t" (.member (.var "tid") "x")
      , .ifNoElse (.bin ">=" (.var "t")
            (.member (.var "params") "num_triangles"))
          [ .ret none ]
      , .declInit uintTy "i0"
          (.call "uint" [.index (.var "tri_indices")
            (.bin "*" (.var "t") (.litUint 3))])
      , .declInit uintTy "i1"
          (.call "uint" [.index (.var "tri_indices")
            (.bin "+" (.bin "*" (.var "t") (.litUint 3)) (.litUint 1))])
      , .declInit uintTy "i2"
          (.call "uint" [.index (.var "tri_indices")
            (.bin "+" (.bin "*" (.var "t") (.litUint 3)) (.litUint 2))])
      , .declInit f3Ty "a" (.index (.var "positions") (.var "i0"))
      , .declInit f3Ty "b" (.index (.var "positions") (.var "i1"))
      , .declInit f3Ty "c" (.index (.var "positions") (.var "i2"))
      , .declInit floatTy "cot_a"
          (.call "cot_at_apex" [.var "a", .var "b", .var "c"])
      , .declInit floatTy "cot_b"
          (.call "cot_at_apex" [.var "b", .var "a", .var "c"])
      , .declInit floatTy "cot_c"
          (.call "cot_at_apex" [.var "c", .var "a", .var "b"])
      , .declInit floatTy "w_bc"
          (.bin "*" (.var "cot_a") (.litFloat 0.5))
      , .declInit floatTy "w_ac"
          (.bin "*" (.var "cot_b") (.litFloat 0.5))
      , .declInit floatTy "w_ab"
          (.bin "*" (.var "cot_c") (.litFloat 0.5))
      , .declInit uintTy "base"
          (.bin "*" (.var "t") (.litUint 9))
      , setM (.var "base") (.litUint 0) (.litUint 0)
          (.bin "+" (.var "w_ab") (.var "w_ac"))
      , setM (.var "base") (.litUint 0) (.litUint 1) (.un "-" (.var "w_ab"))
      , setM (.var "base") (.litUint 0) (.litUint 2) (.un "-" (.var "w_ac"))
      , setM (.var "base") (.litUint 1) (.litUint 0) (.un "-" (.var "w_ab"))
      , setM (.var "base") (.litUint 1) (.litUint 1)
          (.bin "+" (.var "w_ab") (.var "w_bc"))
      , setM (.var "base") (.litUint 1) (.litUint 2) (.un "-" (.var "w_bc"))
      , setM (.var "base") (.litUint 2) (.litUint 0) (.un "-" (.var "w_ac"))
      , setM (.var "base") (.litUint 2) (.litUint 1) (.un "-" (.var "w_bc"))
      , setM (.var "base") (.litUint 2) (.litUint 2)
          (.bin "+" (.var "w_ac") (.var "w_bc"))
      , .ret none ] }

def shader : SlangShaderModule :=
  { structs :=
      [ { name := "PolyLapParams"
        , fields :=
            [ ⟨"num_triangles", .scalar .uint, Semantic.none, none, none, .qIn⟩ ] } ]
  , globals :=
      [ ⟨"params",      .const "PolyLapParams",      Semantic.none, some 0, some 0, .qIn⟩
      , ⟨"positions",   .roBuf f3Ty,                 Semantic.none, some 1, some 0, .qIn⟩
      , ⟨"tri_indices", .roBuf (.scalar .int),       Semantic.none, some 2, some 0, .qIn⟩
      , ⟨"matrices",    .rwBuf (.scalar .float),     Semantic.none, some 3, some 0, .qIn⟩ ]
  , functions := [cot_at_apex, mainEntry] }

def expected : String :=
"struct PolyLapParams {
  uint num_triangles;
};

[[vk::binding(0, 0)]]
ConstantBuffer<PolyLapParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<float3> positions;
[[vk::binding(2, 0)]]
StructuredBuffer<int> tri_indices;
[[vk::binding(3, 0)]]
RWStructuredBuffer<float> matrices;

float cot_at_apex(float3 apex, float3 o1, float3 o2) {
  float3 a = (o1 - apex);
  float3 b = (o2 - apex);
  float d = dot(a, b);
  float cl = length(cross(a, b));
  return (d / cl);
}

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint t = tid.x;
  if ((t >= params.num_triangles)) {
    return;
  }
  uint i0 = uint(tri_indices[(t * 3u)]);
  uint i1 = uint(tri_indices[((t * 3u) + 1u)]);
  uint i2 = uint(tri_indices[((t * 3u) + 2u)]);
  float3 a = positions[i0];
  float3 b = positions[i1];
  float3 c = positions[i2];
  float cot_a = cot_at_apex(a, b, c);
  float cot_b = cot_at_apex(b, a, c);
  float cot_c = cot_at_apex(c, a, b);
  float w_bc = (cot_a * 0.500000);
  float w_ac = (cot_b * 0.500000);
  float w_ab = (cot_c * 0.500000);
  uint base = (t * 9u);
  matrices[(base + ((0u * 3u) + 0u))] = (w_ab + w_ac);
  matrices[(base + ((0u * 3u) + 1u))] = (-w_ab);
  matrices[(base + ((0u * 3u) + 2u))] = (-w_ac);
  matrices[(base + ((1u * 3u) + 0u))] = (-w_ab);
  matrices[(base + ((1u * 3u) + 1u))] = (w_ab + w_bc);
  matrices[(base + ((1u * 3u) + 2u))] = (-w_bc);
  matrices[(base + ((2u * 3u) + 0u))] = (-w_ac);
  matrices[(base + ((2u * 3u) + 1u))] = (-w_bc);
  matrices[(base + ((2u * 3u) + 2u))] = (w_ac + w_bc);
  return;
}"

example : LeanSlang.emit shader = expected := by native_decide

example : shader.entryPointName = "main" := by native_decide

end Curvenet.SlangCodegen.PolygonLaplacian
