import LeanSlang
import Curvenet.SlangCodegen.Common

/-!
# `Curvenet.SlangCodegen.SurfaceProjection` — per-vertex closest-triangle

Slang re-port of `src/curvenet/surface_projection.h`. Per query
point, finds the closest triangle on the rest mesh and writes the
projected position + the triangle index. No BVH (yet) — linear scan
over all triangles, O(V·F). Adequate at the curvenet scale (handfuls
of curves, hundreds of segments); a BVH path can replace this loop
when V·F crosses ~1e7.

Bindings:
  0 — `ConstantBuffer<SurfProjParams> { uint num_queries; uint num_tris; }`
  1 — `StructuredBuffer<float3> queries`     (length num_queries)
  2 — `StructuredBuffer<float3> tri_a`       (length num_tris)
  3 — `StructuredBuffer<float3> tri_b`       (length num_tris)
  4 — `StructuredBuffer<float3> tri_c`       (length num_tris)
  5 — `RWStructuredBuffer<float3> projected` (length num_queries)
  6 — `RWStructuredBuffer<int>    tri_idx`   (length num_queries)
-/

namespace Curvenet.SlangCodegen.SurfaceProjection

open LeanSlang
open Curvenet.SlangCodegen.Common


/-- Closest point on triangle (a, b, c) to query q, returning the
    projected float3. Implements the standard 7-region case split
    (Möller-Aila) in compact form. -/
private def closest_on_tri : SlangFunctionDecl :=
  { attrs := []
  , retType := f3Ty
  , name := "closest_on_tri"
  , params :=
      [ ⟨"q", f3Ty, Semantic.none, none, none, .qIn⟩
      , ⟨"a", f3Ty, Semantic.none, none, none, .qIn⟩
      , ⟨"b", f3Ty, Semantic.none, none, none, .qIn⟩
      , ⟨"c", f3Ty, Semantic.none, none, none, .qIn⟩ ]
  , body :=
      [ .declInit f3Ty "ab" (.bin "-" (.var "b") (.var "a"))
      , .declInit f3Ty "ac" (.bin "-" (.var "c") (.var "a"))
      , .declInit f3Ty "ap" (.bin "-" (.var "q") (.var "a"))
      , .declInit floatTy "d1" (.call "dot" [.var "ab", .var "ap"])
      , .declInit floatTy "d2" (.call "dot" [.var "ac", .var "ap"])
      , .ifNoElse
          (.bin "&&" (.bin "<=" (.var "d1") (.litFloat 0.0))
                      (.bin "<=" (.var "d2") (.litFloat 0.0)))
          [ .retExpr (.var "a") ]
      , .declInit f3Ty "bp" (.bin "-" (.var "q") (.var "b"))
      , .declInit floatTy "d3" (.call "dot" [.var "ab", .var "bp"])
      , .declInit floatTy "d4" (.call "dot" [.var "ac", .var "bp"])
      , .ifNoElse
          (.bin "&&" (.bin ">=" (.var "d3") (.litFloat 0.0))
                      (.bin "<=" (.var "d4") (.var "d3")))
          [ .retExpr (.var "b") ]
      , .declInit f3Ty "cp" (.bin "-" (.var "q") (.var "c"))
      , .declInit floatTy "d5" (.call "dot" [.var "ab", .var "cp"])
      , .declInit floatTy "d6" (.call "dot" [.var "ac", .var "cp"])
      , .ifNoElse
          (.bin "&&" (.bin ">=" (.var "d6") (.litFloat 0.0))
                      (.bin "<=" (.var "d5") (.var "d6")))
          [ .retExpr (.var "c") ]
      , -- Interior projection: P = A + u*ab + v*ac with barycentric solve
        .declInit floatTy "vc"
          (.bin "-" (.bin "*" (.var "d1") (.var "d4"))
                    (.bin "*" (.var "d3") (.var "d2")))
      , .ifNoElse
          (.bin "&&"
            (.bin "&&" (.bin "<=" (.var "vc") (.litFloat 0.0))
                        (.bin ">=" (.var "d1") (.litFloat 0.0)))
            (.bin "<=" (.var "d3") (.litFloat 0.0)))
          [ .declInit floatTy "v_e" (.bin "/" (.var "d1")
              (.bin "-" (.var "d1") (.var "d3")))
          , .retExpr
              (.bin "+" (.var "a")
                (.bin "*" (.var "v_e") (.var "ab"))) ]
      , -- Fall through to barycentric interior
        .declInit floatTy "denom"
          (.bin "+" (.var "vc")
            (.bin "+"
              (.bin "*" (.var "d4")
                (.bin "-" (.var "d6") (.var "d4")))
              (.bin "*" (.var "d5")
                (.bin "-" (.var "d2") (.var "d5")))))
      , .declInit floatTy "v"
          (.ternary (.bin "<" (.call "abs" [.var "denom"]) (.litFloat 1e-30))
            (.litFloat 0.0)
            (.bin "/" (.var "vc") (.var "denom")))
      , .declInit floatTy "w"
          (.ternary (.bin "<" (.call "abs" [.var "denom"]) (.litFloat 1e-30))
            (.litFloat 0.0)
            (.bin "/"
              (.bin "-" (.bin "*" (.var "d1") (.var "d4"))
                        (.bin "*" (.var "d3") (.var "d2")))
              (.var "denom")))
      , .retExpr
          (.bin "+" (.var "a")
            (.bin "+"
              (.bin "*" (.var "v") (.var "ab"))
              (.bin "*" (.var "w") (.var "ac")))) ] }

private def mainEntry : SlangFunctionDecl :=
  { attrs  := [.shaderCompute, .numthreads 64 1 1]
  , name   := "main"
  , params := [⟨"tid", .vec .uint 3, .svDispatchThreadId, none, none, .qIn⟩]
  , body   :=
      [ .declInit uintTy "qi" (.member (.var "tid") "x")
      , .ifNoElse (.bin ">=" (.var "qi") (.member (.var "params") "num_queries"))
          [ .ret none ]
      , .declInit f3Ty "q" (.index (.var "queries") (.var "qi"))
      , .declInit f3Ty "best_p"
          (.call "float3" [.litFloat 0.0, .litFloat 0.0, .litFloat 0.0])
      , .declInit floatTy "best_d2" (.litFloat 1e30)
      , .declInit intTy "best_t" (.un "-" (.litUint 1))
      , .forCount "t" (.litUint 0) (.member (.var "params") "num_tris")
          [ .declInit f3Ty "p"
              (.call "closest_on_tri"
                [.var "q"
                , .index (.var "tri_a") (.var "t")
                , .index (.var "tri_b") (.var "t")
                , .index (.var "tri_c") (.var "t")])
          , .declInit f3Ty "v" (.bin "-" (.var "p") (.var "q"))
          , .declInit floatTy "d2" (.call "dot" [.var "v", .var "v"])
          , .ifNoElse (.bin "<" (.var "d2") (.var "best_d2"))
              [ .assign (.var "best_d2") (.var "d2")
              , .assign (.var "best_p") (.var "p")
              , .assign (.var "best_t") (.call "int" [.var "t"]) ] ]
      , .assign (.index (.var "projected") (.var "qi")) (.var "best_p")
      , .assign (.index (.var "tri_idx") (.var "qi")) (.var "best_t")
      , .ret none ] }

def shader : SlangShaderModule :=
  { structs :=
      [ { name := "SurfProjParams"
        , fields :=
            [ ⟨"num_queries", .scalar .uint, Semantic.none, none, none, .qIn⟩
            , ⟨"num_tris",    .scalar .uint, Semantic.none, none, none, .qIn⟩ ] } ]
  , globals :=
      [ ⟨"params",    .const "SurfProjParams", Semantic.none, some 0, some 0, .qIn⟩
      , ⟨"queries",   .roBuf f3Ty,             Semantic.none, some 1, some 0, .qIn⟩
      , ⟨"tri_a",     .roBuf f3Ty,             Semantic.none, some 2, some 0, .qIn⟩
      , ⟨"tri_b",     .roBuf f3Ty,             Semantic.none, some 3, some 0, .qIn⟩
      , ⟨"tri_c",     .roBuf f3Ty,             Semantic.none, some 4, some 0, .qIn⟩
      , ⟨"projected", .rwBuf f3Ty,             Semantic.none, some 5, some 0, .qIn⟩
      , ⟨"tri_idx",   .rwBuf intTy,            Semantic.none, some 6, some 0, .qIn⟩ ]
  , functions := [closest_on_tri, mainEntry] }

def expected : String :=
"struct SurfProjParams {
  uint num_queries;
  uint num_tris;
};

[[vk::binding(0, 0)]]
ConstantBuffer<SurfProjParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<float3> queries;
[[vk::binding(2, 0)]]
StructuredBuffer<float3> tri_a;
[[vk::binding(3, 0)]]
StructuredBuffer<float3> tri_b;
[[vk::binding(4, 0)]]
StructuredBuffer<float3> tri_c;
[[vk::binding(5, 0)]]
RWStructuredBuffer<float3> projected;
[[vk::binding(6, 0)]]
RWStructuredBuffer<int> tri_idx;

float3 closest_on_tri(float3 q, float3 a, float3 b, float3 c) {
  float3 ab = (b - a);
  float3 ac = (c - a);
  float3 ap = (q - a);
  float d1 = dot(ab, ap);
  float d2 = dot(ac, ap);
  if (((d1 <= 0.000000) && (d2 <= 0.000000))) {
    return a;
  }
  float3 bp = (q - b);
  float d3 = dot(ab, bp);
  float d4 = dot(ac, bp);
  if (((d3 >= 0.000000) && (d4 <= d3))) {
    return b;
  }
  float3 cp = (q - c);
  float d5 = dot(ab, cp);
  float d6 = dot(ac, cp);
  if (((d6 >= 0.000000) && (d5 <= d6))) {
    return c;
  }
  float vc = ((d1 * d4) - (d3 * d2));
  if ((((vc <= 0.000000) && (d1 >= 0.000000)) && (d3 <= 0.000000))) {
    float v_e = (d1 / (d1 - d3));
    return (a + (v_e * ab));
  }
  float denom = (vc + ((d4 * (d6 - d4)) + (d5 * (d2 - d5))));
  float v = ((abs(denom) < 0.000000) ? 0.000000 : (vc / denom));
  float w = ((abs(denom) < 0.000000) ? 0.000000 : (((d1 * d4) - (d3 * d2)) / denom));
  return (a + ((v * ab) + (w * ac)));
}

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint qi = tid.x;
  if ((qi >= params.num_queries)) {
    return;
  }
  float3 q = queries[qi];
  float3 best_p = float3(0.000000, 0.000000, 0.000000);
  float best_d2 = 1000000000000000019884624838656.000000;
  int best_t = (-1u);
  for (uint t = 0u; t < params.num_tris; ++t) {
    float3 p = closest_on_tri(q, tri_a[t], tri_b[t], tri_c[t]);
    float3 v = (p - q);
    float d2 = dot(v, v);
    if ((d2 < best_d2)) {
      best_d2 = d2;
      best_p = p;
      best_t = int(t);
    }
  }
  projected[qi] = best_p;
  tri_idx[qi] = best_t;
  return;
}"

example : LeanSlang.emit shader = expected := by native_decide

example : shader.entryPointName = "main" := by native_decide

end Curvenet.SlangCodegen.SurfaceProjection
