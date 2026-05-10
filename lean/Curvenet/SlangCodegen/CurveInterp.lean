import LeanSlang

/-!
# `Curvenet.SlangCodegen.CurveInterp` — per-vertex curve lerp

Slang re-port of `src/curvenet/curve_interp.h`. Per-segment linear
blend between two intersection-supplied (normal, width) endpoints.
Output index `i` carries `lerp(first, last, i / (n - 1))` for both
the SideData fields (3-vec normal + scalar width).

Bindings (set 0):
  0 — `ConstantBuffer<CurveInterpParams> { uint n; }`  (segment count, ≥ 2)
  1 — `StructuredBuffer<float3> first_n_pair`  (length 2: first.first.n, first.second.n)
  2 — `StructuredBuffer<float3> last_n_pair`   (length 2)
  3 — `StructuredBuffer<float>  first_w_pair`  (length 2)
  4 — `StructuredBuffer<float>  last_w_pair`   (length 2)
  5 — `RWStructuredBuffer<float3> out_n` (length 2 * n: per-side normals)
  6 — `RWStructuredBuffer<float>  out_w` (length 2 * n: per-side widths)
-/

namespace Curvenet.SlangCodegen.CurveInterp

open LeanSlang

private def floatTy : SlangType := .scalar .float
private def f3Ty    : SlangType := .vec .float 3
private def uintTy  : SlangType := .scalar .uint

def shader : SlangShaderModule :=
  { structs :=
      [ { name := "CurveInterpParams"
        , fields := [⟨"n", .scalar .uint, Semantic.none, none, none, .qIn⟩] } ]
  , globals :=
      [ ⟨"params",       .const "CurveInterpParams", Semantic.none, some 0, some 0, .qIn⟩
      , ⟨"first_n_pair", .roBuf f3Ty,                Semantic.none, some 1, some 0, .qIn⟩
      , ⟨"last_n_pair",  .roBuf f3Ty,                Semantic.none, some 2, some 0, .qIn⟩
      , ⟨"first_w_pair", .roBuf floatTy,             Semantic.none, some 3, some 0, .qIn⟩
      , ⟨"last_w_pair",  .roBuf floatTy,             Semantic.none, some 4, some 0, .qIn⟩
      , ⟨"out_n",        .rwBuf f3Ty,                Semantic.none, some 5, some 0, .qIn⟩
      , ⟨"out_w",        .rwBuf floatTy,             Semantic.none, some 6, some 0, .qIn⟩ ]
  , functions := [{
      attrs  := [.shaderCompute, .numthreads 64 1 1]
      name   := "main"
      params := [⟨"tid", .vec .uint 3, .svDispatchThreadId, none, none, .qIn⟩]
      body   :=
        [ .declInit uintTy "i" (.member (.var "tid") "x")
        , .ifNoElse (.bin ">=" (.var "i") (.member (.var "params") "n"))
            [ .ret none ]
        , .declInit floatTy "denom"
            (.call "max"
              [ .litFloat 1.0
              , .call "float"
                  [ .bin "-" (.member (.var "params") "n") (.litUint 1) ] ])
        , .declInit floatTy "t"
            (.bin "/" (.call "float" [.var "i"]) (.var "denom"))
        , .declInit floatTy "omt" (.bin "-" (.litFloat 1.0) (.var "t"))
        , -- Side 0
          .assign (.index (.var "out_n") (.bin "*" (.litUint 2) (.var "i")))
            (.bin "+"
              (.bin "*" (.var "omt") (.index (.var "first_n_pair") (.litUint 0)))
              (.bin "*" (.var "t")   (.index (.var "last_n_pair")  (.litUint 0))))
        , .assign (.index (.var "out_w") (.bin "*" (.litUint 2) (.var "i")))
            (.bin "+"
              (.bin "*" (.var "omt") (.index (.var "first_w_pair") (.litUint 0)))
              (.bin "*" (.var "t")   (.index (.var "last_w_pair")  (.litUint 0))))
        , -- Side 1
          .assign (.index (.var "out_n")
              (.bin "+" (.bin "*" (.litUint 2) (.var "i")) (.litUint 1)))
            (.bin "+"
              (.bin "*" (.var "omt") (.index (.var "first_n_pair") (.litUint 1)))
              (.bin "*" (.var "t")   (.index (.var "last_n_pair")  (.litUint 1))))
        , .assign (.index (.var "out_w")
              (.bin "+" (.bin "*" (.litUint 2) (.var "i")) (.litUint 1)))
            (.bin "+"
              (.bin "*" (.var "omt") (.index (.var "first_w_pair") (.litUint 1)))
              (.bin "*" (.var "t")   (.index (.var "last_w_pair")  (.litUint 1))))
        , .ret none ] }] }

example : shader.entryPointName = "main" := by native_decide

end Curvenet.SlangCodegen.CurveInterp
