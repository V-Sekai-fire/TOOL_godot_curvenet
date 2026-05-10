import LeanSlang

/-!
# `Curvenet.SlangCodegen.ScaledFrames` — per-segment deformation gradient

Slang re-port of `src/curvenet/scaled_frames.h`, focusing on the
DeGoes22 §3 deformation-gradient ratio
`F_i = (B_i S_i) · (B̆_i S̆_i)^{-1}` between current and rest
configurations. Per-segment input pair (rest_frame, posed_frame) →
output F. One thread per segment.

The 3×3 inverse is computed by the cofactor formula inline; degenerate
rest_frames (det = 0) map to the zero matrix to avoid NaN propagation
into the harmonic solve. The `smallest_rotation` /
`isolated_segment_gradient` helpers from the C++ header live in their
own submodule alongside (segment_gradient).

Bindings (set 0):
  0 — `ConstantBuffer<ScaledFramesParams> { uint num_segments; }`
  1 — `StructuredBuffer<float> rest_frames`   (9 floats per segment, row-major 3×3)
  2 — `StructuredBuffer<float> posed_frames`  (9 floats per segment, row-major 3×3)
  3 — `RWStructuredBuffer<float> F`           (9 floats per segment, output)
-/

namespace Curvenet.SlangCodegen.ScaledFrames

open LeanSlang

private def floatTy : SlangType := .scalar .float
private def m3Ty    : SlangType := .mat .float 3 3
private def uintTy  : SlangType := .scalar .uint

/-- `loadM3(buf, base)` packs 9 contiguous floats at `base` into a
    float3x3. Returns by writing into 9 named locals. We expose the
    emitted output as a function call so the main entry can be tight. -/
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

/-- 3×3 inverse via cofactor expansion. Returns the zero matrix when
    `|det| < 1e-30` to avoid NaN propagation. -/
private def inv3x3 : SlangFunctionDecl :=
  { attrs := []
  , retType := m3Ty
  , name := "inv3x3"
  , params := [⟨"m", m3Ty, Semantic.none, none, none, .qIn⟩]
  , body :=
      [ -- Extract 9 entries via _mNN swizzle (Slang convention).
        .declInit floatTy "a" (.member (.var "m") "_m00")
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
          [ .bin "*" (.var "invDet")
              (.bin "-" (.bin "*" (.var "e") (.var "i"))
                        (.bin "*" (.var "f") (.var "h")))
          , .bin "*" (.var "invDet")
              (.bin "-" (.bin "*" (.var "c") (.var "h"))
                        (.bin "*" (.var "b") (.var "i")))
          , .bin "*" (.var "invDet")
              (.bin "-" (.bin "*" (.var "b") (.var "f"))
                        (.bin "*" (.var "c") (.var "e")))
          , .bin "*" (.var "invDet")
              (.bin "-" (.bin "*" (.var "f") (.var "g"))
                        (.bin "*" (.var "d") (.var "i")))
          , .bin "*" (.var "invDet")
              (.bin "-" (.bin "*" (.var "a") (.var "i"))
                        (.bin "*" (.var "c") (.var "g")))
          , .bin "*" (.var "invDet")
              (.bin "-" (.bin "*" (.var "c") (.var "d"))
                        (.bin "*" (.var "a") (.var "f")))
          , .bin "*" (.var "invDet")
              (.bin "-" (.bin "*" (.var "d") (.var "h"))
                        (.bin "*" (.var "e") (.var "g")))
          , .bin "*" (.var "invDet")
              (.bin "-" (.bin "*" (.var "b") (.var "g"))
                        (.bin "*" (.var "a") (.var "h")))
          , .bin "*" (.var "invDet")
              (.bin "-" (.bin "*" (.var "a") (.var "e"))
                        (.bin "*" (.var "b") (.var "d"))) ]) ] }

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
      , .ifNoElse (.bin ">=" (.var "t")
            (.member (.var "params") "num_segments"))
          [ .ret none ]
      , .declInit uintTy "base" (.bin "*" (.var "t") (.litUint 9))
      , .declInit m3Ty "rest"
          (.call "loadM3" [.var "rest_frames", .var "base"])
      , .declInit m3Ty "posed"
          (.call "loadM3" [.var "posed_frames", .var "base"])
      , .declInit m3Ty "rest_inv"
          (.call "inv3x3" [.var "rest"])
      , .declInit m3Ty "Fmat"
          (.call "mul" [.var "posed", .var "rest_inv"])
      ] ++ storeM3 (.var "base") (.var "Fmat") ++
      [ .ret none ] }

def shader : SlangShaderModule :=
  { structs :=
      [ { name := "ScaledFramesParams"
        , fields :=
            [ ⟨"num_segments", .scalar .uint, Semantic.none, none, none, .qIn⟩ ] } ]
  , globals :=
      [ ⟨"params",       .const "ScaledFramesParams", Semantic.none, some 0, some 0, .qIn⟩
      , ⟨"rest_frames",  .roBuf floatTy,              Semantic.none, some 1, some 0, .qIn⟩
      , ⟨"posed_frames", .roBuf floatTy,              Semantic.none, some 2, some 0, .qIn⟩
      , ⟨"F",            .rwBuf floatTy,              Semantic.none, some 3, some 0, .qIn⟩ ]
  , functions := [loadM3, inv3x3, mainEntry] }

example : shader.entryPointName = "main" := by native_decide

end Curvenet.SlangCodegen.ScaledFrames
