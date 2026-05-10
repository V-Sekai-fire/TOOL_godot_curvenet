import LeanSlang

/-!
# `Curvenet.SlangCodegen.DenseLinAlg` — small-system dense solve

Slang re-port of the dense LU path in `src/curvenet/dense_linalg.h`.
Solves an N×N dense system `A x = b` with `N ≤ 16` (curvenet sample
matrices are tiny). One thread per system; the host packs many
independent systems for the same dispatch (e.g. per-handle bind-time
unit-perturbation solves).

LU factorization without pivoting; OK for the curvenet bind-time
SPD systems but not safe for general dense — production callers
verify `det != 0` before dispatch.

Bindings (set 0):
  0 — `ConstantBuffer<DenseLinAlgParams> { uint num_systems; uint n; }`
  1 — `StructuredBuffer<float> A` (n*n floats per system)
  2 — `StructuredBuffer<float> b` (n floats per system)
  3 — `RWStructuredBuffer<float> x` (n floats per system, output)

A `N_MAX = 16` stack array is used for the working LU. Set the
shader's `N_MAX` literal to match the host's max dimension.
-/

namespace Curvenet.SlangCodegen.DenseLinAlg

open LeanSlang

private def floatTy : SlangType := .scalar .float
private def uintTy  : SlangType := .scalar .uint

def shader : SlangShaderModule :=
  { structs :=
      [ { name := "DenseLinAlgParams"
        , fields :=
            [ ⟨"num_systems", .scalar .uint, Semantic.none, none, none, .qIn⟩
            , ⟨"n",           .scalar .uint, Semantic.none, none, none, .qIn⟩ ] } ]
  , globals :=
      [ ⟨"params", .const "DenseLinAlgParams", Semantic.none, some 0, some 0, .qIn⟩
      , ⟨"A",      .roBuf floatTy,             Semantic.none, some 1, some 0, .qIn⟩
      , ⟨"b",      .roBuf floatTy,             Semantic.none, some 2, some 0, .qIn⟩
      , ⟨"x",      .rwBuf floatTy,             Semantic.none, some 3, some 0, .qIn⟩ ]
  , functions := [{
      attrs  := [.shaderCompute, .numthreads 64 1 1]
      name   := "main"
      params := [⟨"tid", .vec .uint 3, .svDispatchThreadId, none, none, .qIn⟩]
      body   :=
        [ .declInit uintTy "s" (.member (.var "tid") "x")
        , .ifNoElse (.bin ">=" (.var "s") (.member (.var "params") "num_systems"))
            [ .ret none ]
        , .declInit uintTy "n" (.member (.var "params") "n")
        , .declInit uintTy "abase" (.bin "*" (.var "s") (.bin "*" (.var "n") (.var "n")))
        , .declInit uintTy "vbase" (.bin "*" (.var "s") (.var "n"))
        , .declareArray floatTy "M" 256  -- N_MAX² stack scratch
        , .declareArray floatTy "y" 16
        , -- Load A into M
          .forCount "i" (.litUint 0) (.var "n")
            [ .forCount "j" (.litUint 0) (.var "n")
                [ .assign
                    (.index (.var "M")
                      (.bin "+"
                        (.bin "*" (.var "i") (.var "n"))
                        (.var "j")))
                    (.index (.var "A")
                      (.bin "+" (.var "abase")
                        (.bin "+"
                          (.bin "*" (.var "i") (.var "n"))
                          (.var "j")))) ] ]
        , -- LU without pivoting: Doolittle in-place.
          .forCount "k" (.litUint 0) (.var "n")
            [ .declInit floatTy "akk"
                (.index (.var "M")
                  (.bin "+"
                    (.bin "*" (.var "k") (.var "n"))
                    (.var "k")))
            , .ifNoElse (.bin ">" (.call "abs" [.var "akk"]) (.litFloat 1e-30))
                [ .forCount "i" (.bin "+" (.var "k") (.litUint 1)) (.var "n")
                    [ .declInit floatTy "factor"
                        (.bin "/"
                          (.index (.var "M")
                            (.bin "+"
                              (.bin "*" (.var "i") (.var "n"))
                              (.var "k")))
                          (.var "akk"))
                    , .assign
                        (.index (.var "M")
                          (.bin "+"
                            (.bin "*" (.var "i") (.var "n"))
                            (.var "k")))
                        (.var "factor")
                    , .forCount "j" (.bin "+" (.var "k") (.litUint 1)) (.var "n")
                        [ .assign
                            (.index (.var "M")
                              (.bin "+"
                                (.bin "*" (.var "i") (.var "n"))
                                (.var "j")))
                            (.bin "-"
                              (.index (.var "M")
                                (.bin "+"
                                  (.bin "*" (.var "i") (.var "n"))
                                  (.var "j")))
                              (.bin "*" (.var "factor")
                                (.index (.var "M")
                                  (.bin "+"
                                    (.bin "*" (.var "k") (.var "n"))
                                    (.var "j")))))
                        ] ] ] ]
        , -- Forward solve  L y = b   (L is unit-lower in M)
          .forCount "i" (.litUint 0) (.var "n")
            [ .declInit floatTy "yi"
                (.index (.var "b") (.bin "+" (.var "vbase") (.var "i")))
            , .forCount "j" (.litUint 0) (.var "i")
                [ .assign (.var "yi")
                    (.bin "-" (.var "yi")
                      (.bin "*"
                        (.index (.var "M")
                          (.bin "+"
                            (.bin "*" (.var "i") (.var "n"))
                            (.var "j")))
                        (.index (.var "y") (.var "j")))) ]
            , .assign (.index (.var "y") (.var "i")) (.var "yi") ]
        , -- Back solve  U x = y
          .forCount "ii" (.litUint 0) (.var "n")
            [ .declInit uintTy "i"
                (.bin "-" (.bin "-" (.var "n") (.litUint 1)) (.var "ii"))
            , .declInit floatTy "xi" (.index (.var "y") (.var "i"))
            , .forCount "j" (.bin "+" (.var "i") (.litUint 1)) (.var "n")
                [ .assign (.var "xi")
                    (.bin "-" (.var "xi")
                      (.bin "*"
                        (.index (.var "M")
                          (.bin "+"
                            (.bin "*" (.var "i") (.var "n"))
                            (.var "j")))
                        (.index (.var "x")
                          (.bin "+" (.var "vbase") (.var "j"))))) ]
            , .declInit floatTy "uii"
                (.index (.var "M")
                  (.bin "+"
                    (.bin "*" (.var "i") (.var "n"))
                    (.var "i")))
            , .assign (.index (.var "x") (.bin "+" (.var "vbase") (.var "i")))
                (.ternary (.bin "<" (.call "abs" [.var "uii"]) (.litFloat 1e-30))
                  (.litFloat 0.0)
                  (.bin "/" (.var "xi") (.var "uii"))) ]
        , .ret none ] }] }

example : shader.entryPointName = "main" := by native_decide

end Curvenet.SlangCodegen.DenseLinAlg
