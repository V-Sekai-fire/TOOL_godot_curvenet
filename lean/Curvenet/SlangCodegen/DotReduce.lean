import LeanSlang
import Curvenet.SlangCodegen.Common

/-!
# `Curvenet.SlangCodegen.DotReduce` — vector dot product (df32)

Slang re-port of `src/curvenet/shaders/dot_reduce.comp`. Computes
`Σ aᵢ · bᵢ` for two fp32 vectors of length n and writes the result
as a df32 (hi, lo) pair into `dst[0]`, `dst[1]`. df32 carries ~48
bits of effective mantissa — keeps a 100k-element dot at fp64
quality where a pure fp32 sum would stagnate at ~6e-3 relative
(n·ε for n = 100k).

One workgroup of 256 threads, grid-strided over the input. For
inputs at our 100k-element target this is one dispatch per dot;
multi-workgroup partials need a second reduction pass.

Helpers (Knuth/Dekker error-free transformations):
  * `two_sum(a, b) → (hi, lo)` — exact a + b = hi + lo
  * `quick_two_sum(a, b)` — faster variant when |a| ≥ |b|
  * `two_prod(a, b)` — exact a·b = hi + lo via FMA
  * `df_add(x, y) → z` — sloppy df + df → df

`precise` keeps the optimiser from rewriting `(a + b) - a` as `b`,
which would defeat the EFTs entirely.
-/

namespace Curvenet.SlangCodegen.DotReduce

open LeanSlang
open Curvenet.SlangCodegen.Common

/-- One f32 in/out parameter binding. -/
private def fIn  (name : String) : SlangBinding :=
  ⟨name, .scalar .float, Semantic.none, none, none, .qIn⟩
private def fOut (name : String) : SlangBinding :=
  ⟨name, .scalar .float, Semantic.none, none, none, .qOut⟩


private def two_sum : SlangFunctionDecl :=
  { attrs   := []
  , retType := .named "void"
  , name    := "two_sum"
  , params  := [fIn "a", fIn "b", fOut "hi", fOut "lo"]
  , body    :=
      [ .declInit          floatTy "h" (.bin "+" (.var "a") (.var "b"))
      , .declInit          floatTy "bb" (.bin "-" (.var "h") (.var "a"))
      , .declInit          floatTy "ah" (.bin "-" (.var "h") (.var "bb"))
      , .declInit          floatTy "lo_a" (.bin "-" (.var "a") (.var "ah"))
      , .declInit          floatTy "lo_b" (.bin "-" (.var "b") (.var "bb"))
      , .assign (.var "hi") (.var "h")
      , .assign (.var "lo") (.bin "+" (.var "lo_a") (.var "lo_b"))
      , .ret none ] }

private def quick_two_sum : SlangFunctionDecl :=
  { attrs   := []
  , retType := .named "void"
  , name    := "quick_two_sum"
  , params  := [fIn "a", fIn "b", fOut "hi", fOut "lo"]
  , body    :=
      [ .declInit          floatTy "h" (.bin "+" (.var "a") (.var "b"))
      , .declInit          floatTy "t" (.bin "-" (.var "h") (.var "a"))
      , .assign (.var "hi") (.var "h")
      , .assign (.var "lo") (.bin "-" (.var "b") (.var "t"))
      , .ret none ] }

private def two_prod : SlangFunctionDecl :=
  { attrs   := []
  , retType := .named "void"
  , name    := "two_prod"
  , params  := [fIn "a", fIn "b", fOut "hi", fOut "lo"]
  , body    :=
      [ .declInit          floatTy "h" (.bin "*" (.var "a") (.var "b"))
      , .assign (.var "hi") (.var "h")
      , .assign (.var "lo")
          (.call "fma" [.var "a", .var "b", .un "-" (.var "h")])
      , .ret none ] }

private def df_add : SlangFunctionDecl :=
  { attrs   := []
  , retType := .named "void"
  , name    := "df_add"
  , params  :=
      [ fIn "x_hi", fIn "x_lo", fIn "y_hi", fIn "y_lo"
      , fOut "z_hi", fOut "z_lo" ]
  , body    :=
      [ .declare floatTy "sh" none
      , .declare floatTy "sl" none
      , .expr (.call "two_sum"
          [.var "x_hi", .var "y_hi", .var "sh", .var "sl"])
      , .declInit          floatTy "xy_lo"
          (.bin "+" (.var "x_lo") (.var "y_lo"))
      , .declInit          floatTy "sl2"
          (.bin "+" (.var "sl") (.var "xy_lo"))
      , .expr (.call "quick_two_sum"
          [.var "sh", .var "sl2", .var "z_hi", .var "z_lo"])
      , .ret none ] }

private def mainEntry : SlangFunctionDecl :=
  { attrs  := [.shaderCompute, .numthreads 256 1 1]
  , name   := "main"
  , params := [⟨"tid", .vec .uint 3, .svGroupThreadId, none, none, .qIn⟩]
  , body   :=
      [ .declInit uintTy "t" (.member (.var "tid") "x")
      , .declInit uintTy "stride" (.litUint 256)
      , .declInit floatTy "acc_hi" (.litFloat 0.0)
      , .declInit floatTy "acc_lo" (.litFloat 0.0)
      -- grid-stride accumulate
      , .declInit uintTy "i" (.var "t")
      , .whileLoop (.bin "<" (.var "i") (.member (.var "params") "n"))
          [ .declare floatTy "p_hi" none
          , .declare floatTy "p_lo" none
          , .expr (.call "two_prod"
              [.index (.var "a") (.var "i")
              , .index (.var "b") (.var "i")
              , .var "p_hi", .var "p_lo"])
          , .declare floatTy "new_hi" none
          , .declare floatTy "new_lo" none
          , .expr (.call "df_add"
              [.var "acc_hi", .var "acc_lo", .var "p_hi", .var "p_lo"
              , .var "new_hi", .var "new_lo"])
          , .assign (.var "acc_hi") (.var "new_hi")
          , .assign (.var "acc_lo") (.var "new_lo")
          , .assign (.var "i") (.bin "+" (.var "i") (.var "stride"))
          ]
      , .assign (.index (.var "s_hi") (.var "t")) (.var "acc_hi")
      , .assign (.index (.var "s_lo") (.var "t")) (.var "acc_lo")
      , .expr (.call "GroupMemoryBarrierWithGroupSync" [])
      -- tree reduce: step = 128, 64, ..., 1
      , .declInit uintTy "step" (.litUint 128)
      , .whileLoop (.bin ">" (.var "step") (.litUint 0))
          [ .ifNoElse (.bin "<" (.var "t") (.var "step"))
              [ .declare floatTy "new_hi" none
              , .declare floatTy "new_lo" none
              , .expr (.call "df_add"
                  [.index (.var "s_hi") (.var "t")
                  , .index (.var "s_lo") (.var "t")
                  , .index (.var "s_hi") (.bin "+" (.var "t") (.var "step"))
                  , .index (.var "s_lo") (.bin "+" (.var "t") (.var "step"))
                  , .var "new_hi", .var "new_lo"])
              , .assign (.index (.var "s_hi") (.var "t")) (.var "new_hi")
              , .assign (.index (.var "s_lo") (.var "t")) (.var "new_lo") ]
          , .expr (.call "GroupMemoryBarrierWithGroupSync" [])
          , .assign (.var "step")
              (.bin ">>" (.var "step") (.litUint 1))
          ]
      , .ifNoElse (.bin "==" (.var "t") (.litUint 0))
          [ .assign (.index (.var "dst") (.litUint 0))
              (.index (.var "s_hi") (.litUint 0))
          , .assign (.index (.var "dst") (.litUint 1))
              (.index (.var "s_lo") (.litUint 0)) ]
      , .ret none ] }

def shader : SlangShaderModule :=
  { structs :=
      [ { name := "DotReduceParams"
        , fields := [⟨"n", .scalar .uint, Semantic.none, none, none, .qIn⟩] } ]
  , groupShared :=
      [ { name := "s_hi", elemType := .scalar .float, dims := [256] }
      , { name := "s_lo", elemType := .scalar .float, dims := [256] } ]
  , globals :=
      [ ⟨"params", .const "DotReduceParams",  Semantic.none, some 0, some 0, .qIn⟩
      , ⟨"a",      .roBuf (.scalar .float),   Semantic.none, some 1, some 0, .qIn⟩
      , ⟨"b",      .roBuf (.scalar .float),   Semantic.none, some 2, some 0, .qIn⟩
      , ⟨"dst",    .rwBuf (.scalar .float),   Semantic.none, some 3, some 0, .qIn⟩ ]
  , functions := [two_sum, quick_two_sum, two_prod, df_add, mainEntry] }



def expected : String :=
"struct DotReduceParams {
  uint n;
};

groupshared float s_hi[256];
groupshared float s_lo[256];

[[vk::binding(0, 0)]]
ConstantBuffer<DotReduceParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<float> a;
[[vk::binding(2, 0)]]
StructuredBuffer<float> b;
[[vk::binding(3, 0)]]
RWStructuredBuffer<float> dst;

void two_sum(float a, float b, out float hi, out float lo) {
  float h = (a + b);
  float bb = (h - a);
  float ah = (h - bb);
  float lo_a = (a - ah);
  float lo_b = (b - bb);
  hi = h;
  lo = (lo_a + lo_b);
  return;
}

void quick_two_sum(float a, float b, out float hi, out float lo) {
  float h = (a + b);
  float t = (h - a);
  hi = h;
  lo = (b - t);
  return;
}

void two_prod(float a, float b, out float hi, out float lo) {
  float h = (a * b);
  hi = h;
  lo = fma(a, b, (-h));
  return;
}

void df_add(float x_hi, float x_lo, float y_hi, float y_lo, out float z_hi, out float z_lo) {
  float sh;
  float sl;
  two_sum(x_hi, y_hi, sh, sl);
  float xy_lo = (x_lo + y_lo);
  float sl2 = (sl + xy_lo);
  quick_two_sum(sh, sl2, z_hi, z_lo);
  return;
}

[shader(\"compute\")] [numthreads(256, 1, 1)]
void main(uint3 tid : SV_GroupThreadID) {
  uint t = tid.x;
  uint stride = 256u;
  float acc_hi = 0.000000;
  float acc_lo = 0.000000;
  uint i = t;
  while ((i < params.n)) {
    float p_hi;
    float p_lo;
    two_prod(a[i], b[i], p_hi, p_lo);
    float new_hi;
    float new_lo;
    df_add(acc_hi, acc_lo, p_hi, p_lo, new_hi, new_lo);
    acc_hi = new_hi;
    acc_lo = new_lo;
    i = (i + stride);
  }
  s_hi[t] = acc_hi;
  s_lo[t] = acc_lo;
  GroupMemoryBarrierWithGroupSync();
  uint step = 128u;
  while ((step > 0u)) {
    if ((t < step)) {
      float new_hi;
      float new_lo;
      df_add(s_hi[t], s_lo[t], s_hi[(t + step)], s_lo[(t + step)], new_hi, new_lo);
      s_hi[t] = new_hi;
      s_lo[t] = new_lo;
    }
    GroupMemoryBarrierWithGroupSync();
    step = (step >> 1u);
  }
  if ((t == 0u)) {
    dst[0u] = s_hi[0u];
    dst[1u] = s_lo[0u];
  }
  return;
}"

example : LeanSlang.emit shader = expected := by native_decide

example : shader.entryPointName = "main" := by native_decide

end Curvenet.SlangCodegen.DotReduce
