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

example : shader.entryPointName = "main" := by native_decide

end Curvenet.SlangCodegen.IntersectionFrames
