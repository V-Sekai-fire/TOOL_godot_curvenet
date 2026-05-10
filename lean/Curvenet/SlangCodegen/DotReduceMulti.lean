import LeanSlang

/-!
# `Curvenet.SlangCodegen.DotReduceMulti` — multi-RHS df32 dot reduction

Slang re-port of `src/curvenet/shaders/dot_reduce_multi.comp`. k
separate dot products in one dispatch, each in df32. Output is 2*k
floats — k pairs of (hi, lo) df32. The host folds (double)hi +
(double)lo back to fp64 per column for CG's per-column α and β
scalars.

One workgroup of 128 threads grid-strided over n elements. Each
thread keeps k df32 accumulators in a local stack array (K_MAX=16).
Tree reduces across the 128 lanes, k columns in lockstep.

Workgroup size 128 (vs 256 for single-RHS) keeps shared memory at
k*128*8 bytes ≤ 16KB at K_MAX=16, matching Adreno's per-WG ceiling.
-/

namespace Curvenet.SlangCodegen.DotReduceMulti

open LeanSlang

private def fIn  (name : String) : SlangBinding :=
  ⟨name, .scalar .float, Semantic.none, none, none, .qIn⟩
private def fOut (name : String) : SlangBinding :=
  ⟨name, .scalar .float, Semantic.none, none, none, .qOut⟩

private def floatTy : SlangType := .scalar .float
private def uintTy  : SlangType := .scalar .uint

private def two_sum : SlangFunctionDecl :=
  { attrs := [], retType := .named "void", name := "two_sum"
  , params := [fIn "a", fIn "b", fOut "hi", fOut "lo"]
  , body :=
      [ .declPreciseInit floatTy "h" (.bin "+" (.var "a") (.var "b"))
      , .declPreciseInit floatTy "bb" (.bin "-" (.var "h") (.var "a"))
      , .declPreciseInit floatTy "ah" (.bin "-" (.var "h") (.var "bb"))
      , .declPreciseInit floatTy "lo_a" (.bin "-" (.var "a") (.var "ah"))
      , .declPreciseInit floatTy "lo_b" (.bin "-" (.var "b") (.var "bb"))
      , .assign (.var "hi") (.var "h")
      , .assign (.var "lo") (.bin "+" (.var "lo_a") (.var "lo_b"))
      , .ret none ] }

private def quick_two_sum : SlangFunctionDecl :=
  { attrs := [], retType := .named "void", name := "quick_two_sum"
  , params := [fIn "a", fIn "b", fOut "hi", fOut "lo"]
  , body :=
      [ .declPreciseInit floatTy "h" (.bin "+" (.var "a") (.var "b"))
      , .declPreciseInit floatTy "t" (.bin "-" (.var "h") (.var "a"))
      , .assign (.var "hi") (.var "h")
      , .assign (.var "lo") (.bin "-" (.var "b") (.var "t"))
      , .ret none ] }

private def two_prod : SlangFunctionDecl :=
  { attrs := [], retType := .named "void", name := "two_prod"
  , params := [fIn "a", fIn "b", fOut "hi", fOut "lo"]
  , body :=
      [ .declPreciseInit floatTy "h" (.bin "*" (.var "a") (.var "b"))
      , .assign (.var "hi") (.var "h")
      , .assign (.var "lo")
          (.call "fma" [.var "a", .var "b", .un "-" (.var "h")])
      , .ret none ] }

private def df_add : SlangFunctionDecl :=
  { attrs := [], retType := .named "void", name := "df_add"
  , params :=
      [ fIn "x_hi", fIn "x_lo", fIn "y_hi", fIn "y_lo"
      , fOut "z_hi", fOut "z_lo" ]
  , body :=
      [ .declare floatTy "sh" none
      , .declare floatTy "sl" none
      , .expr (.call "two_sum"
          [.var "x_hi", .var "y_hi", .var "sh", .var "sl"])
      , .declPreciseInit floatTy "xy_lo"
          (.bin "+" (.var "x_lo") (.var "y_lo"))
      , .declPreciseInit floatTy "sl2"
          (.bin "+" (.var "sl") (.var "xy_lo"))
      , .expr (.call "quick_two_sum"
          [.var "sh", .var "sl2", .var "z_hi", .var "z_lo"])
      , .ret none ] }

/-- 2D index: `arr[i][j]`. -/
private def idx2 (arr i j : SlangExpr) : SlangExpr :=
  .index (.index arr i) j

private def mainEntry : SlangFunctionDecl :=
  { attrs  := [.shaderCompute, .numthreads 128 1 1]
  , name   := "main"
  , params := [⟨"tid", .vec .uint 3, .svGroupThreadId, none, none, .qIn⟩]
  , body   :=
      [ .declInit uintTy "t" (.member (.var "tid") "x")
      , .declInit uintTy "stride" (.litUint 128)
      , .declareArray floatTy "acc_hi" 16
      , .declareArray floatTy "acc_lo" 16
      , .forCount "c" (.litUint 0) (.member (.var "params") "k")
          [ .assign (.index (.var "acc_hi") (.var "c")) (.litFloat 0.0)
          , .assign (.index (.var "acc_lo") (.var "c")) (.litFloat 0.0) ]
      , .declInit uintTy "i" (.var "t")
      , .whileLoop (.bin "<" (.var "i") (.member (.var "params") "n"))
          [ .forCount "c" (.litUint 0) (.member (.var "params") "k")
              [ .declare floatTy "p_hi" none
              , .declare floatTy "p_lo" none
              , .expr (.call "two_prod"
                  [ .index (.var "a")
                      (.bin "+"
                        (.bin "*" (.var "i") (.member (.var "params") "k"))
                        (.var "c"))
                  , .index (.var "b")
                      (.bin "+"
                        (.bin "*" (.var "i") (.member (.var "params") "k"))
                        (.var "c"))
                  , .var "p_hi", .var "p_lo" ])
              , .declare floatTy "new_hi" none
              , .declare floatTy "new_lo" none
              , .expr (.call "df_add"
                  [ .index (.var "acc_hi") (.var "c")
                  , .index (.var "acc_lo") (.var "c")
                  , .var "p_hi", .var "p_lo"
                  , .var "new_hi", .var "new_lo" ])
              , .assign (.index (.var "acc_hi") (.var "c")) (.var "new_hi")
              , .assign (.index (.var "acc_lo") (.var "c")) (.var "new_lo") ]
          , .assign (.var "i") (.bin "+" (.var "i") (.var "stride")) ]
      , .forCount "c" (.litUint 0) (.member (.var "params") "k")
          [ .assign (idx2 (.var "s_hi") (.var "t") (.var "c"))
              (.index (.var "acc_hi") (.var "c"))
          , .assign (idx2 (.var "s_lo") (.var "t") (.var "c"))
              (.index (.var "acc_lo") (.var "c")) ]
      , .expr (.call "GroupMemoryBarrierWithGroupSync" [])
      , .declInit uintTy "step" (.litUint 64)
      , .whileLoop (.bin ">" (.var "step") (.litUint 0))
          [ .ifNoElse (.bin "<" (.var "t") (.var "step"))
              [ .forCount "c" (.litUint 0) (.member (.var "params") "k")
                  [ .declare floatTy "new_hi" none
                  , .declare floatTy "new_lo" none
                  , .expr (.call "df_add"
                      [ idx2 (.var "s_hi") (.var "t") (.var "c")
                      , idx2 (.var "s_lo") (.var "t") (.var "c")
                      , idx2 (.var "s_hi")
                          (.bin "+" (.var "t") (.var "step")) (.var "c")
                      , idx2 (.var "s_lo")
                          (.bin "+" (.var "t") (.var "step")) (.var "c")
                      , .var "new_hi", .var "new_lo" ])
                  , .assign (idx2 (.var "s_hi") (.var "t") (.var "c"))
                      (.var "new_hi")
                  , .assign (idx2 (.var "s_lo") (.var "t") (.var "c"))
                      (.var "new_lo") ] ]
          , .expr (.call "GroupMemoryBarrierWithGroupSync" [])
          , .assign (.var "step")
              (.bin ">>" (.var "step") (.litUint 1)) ]
      , .ifNoElse (.bin "==" (.var "t") (.litUint 0))
          [ .forCount "c" (.litUint 0) (.member (.var "params") "k")
              [ .assign (.index (.var "dst")
                  (.bin "+"
                    (.bin "*" (.litUint 2) (.var "c"))
                    (.litUint 0)))
                  (idx2 (.var "s_hi") (.litUint 0) (.var "c"))
              , .assign (.index (.var "dst")
                  (.bin "+"
                    (.bin "*" (.litUint 2) (.var "c"))
                    (.litUint 1)))
                  (idx2 (.var "s_lo") (.litUint 0) (.var "c")) ] ]
      , .ret none ] }

def shader : SlangShaderModule :=
  { structs :=
      [ { name := "DotReduceMultiParams"
        , fields :=
            [ ⟨"n", .scalar .uint, Semantic.none, none, none, .qIn⟩
            , ⟨"k", .scalar .uint, Semantic.none, none, none, .qIn⟩ ] } ]
  , groupShared :=
      [ { name := "s_hi", elemType := .scalar .float, dims := [128, 16] }
      , { name := "s_lo", elemType := .scalar .float, dims := [128, 16] } ]
  , globals :=
      [ ⟨"params", .const "DotReduceMultiParams", Semantic.none, some 0, some 0, .qIn⟩
      , ⟨"a",      .roBuf (.scalar .float),       Semantic.none, some 1, some 0, .qIn⟩
      , ⟨"b",      .roBuf (.scalar .float),       Semantic.none, some 2, some 0, .qIn⟩
      , ⟨"dst",    .rwBuf (.scalar .float),       Semantic.none, some 3, some 0, .qIn⟩ ]
  , functions := [two_sum, quick_two_sum, two_prod, df_add, mainEntry] }

example : shader.entryPointName = "main" := by native_decide

end Curvenet.SlangCodegen.DotReduceMulti
