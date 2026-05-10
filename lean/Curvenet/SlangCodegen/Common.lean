import LeanSlang

/-!
# `Curvenet.SlangCodegen.Common` — shared codegen helpers

Type aliases, parameter-binding shorthands, mat3 storage helpers,
the inline 3×3 inverse, the cotangent triangle weight, and the
Knuth/Dekker error-free transformations. Per-kernel modules import
from here so the AST-construction is in one place.

These are *building blocks* for `SlangShaderModule` values; the
emitted text and binding-set-0 layout are still each kernel's
responsibility, since those vary.
-/

namespace Curvenet.SlangCodegen.Common

open LeanSlang

/-! ## Scalar / vector / matrix type shortcuts -/

def floatTy : SlangType := .scalar .float
def uintTy  : SlangType := .scalar .uint
def intTy   : SlangType := .scalar .int
def boolTy  : SlangType := .scalar .bool
def f3Ty    : SlangType := .vec .float 3
def f4Ty    : SlangType := .vec .float 4
def u3Ty    : SlangType := .vec .uint  3
def m3Ty    : SlangType := .mat .float 3 3
def m4Ty    : SlangType := .mat .float 4 4

/-! ## Parameter binding shorthands -/

/-- `(in)` float function param. -/
def fIn  (name : String) : SlangBinding :=
  ⟨name, floatTy, Semantic.none, none, none, .qIn⟩

/-- `(out)` float function param. -/
def fOut (name : String) : SlangBinding :=
  ⟨name, floatTy, Semantic.none, none, none, .qOut⟩

/-- Standard `tid : SV_DispatchThreadID` entry-point param. -/
def tidParam : SlangBinding :=
  ⟨"tid", u3Ty, Semantic.svDispatchThreadId, none, none, .qIn⟩

/-- Workgroup-local `tid : SV_GroupThreadID` entry-point param. -/
def gtidParam : SlangBinding :=
  ⟨"tid", u3Ty, Semantic.svGroupThreadId, none, none, .qIn⟩

/-- Helper: a global SSBO / cbuffer binding at `[[vk::binding(idx, 0)]]`. -/
def ssbo (idx : Nat) (name : String) (t : SlangType) : SlangBinding :=
  ⟨name, t, Semantic.none, some idx, some 0, .qIn⟩

/-! ## 3×3 matrix helpers -/

/-- `arr[i][j]` — chained index for 2D groupshared arrays etc. -/
def idx2 (arr i j : SlangExpr) : SlangExpr :=
  .index (.index arr i) j

/-- `loadM3(buf, base)` — packs 9 contiguous floats at `[base, base+9)`
    into a row-major float3x3. -/
def loadM3 : SlangFunctionDecl :=
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

/-- Write a float3x3 `m` into 9 contiguous slots of buffer `target`
    at offset `[base, base+9)`, row-major. -/
def storeM3 (target : String) (base : SlangExpr) (m : SlangExpr) : List SlangStmt :=
  [ .assign (.index (.var target) (.bin "+" base (.litUint 0))) (.member m "_m00")
  , .assign (.index (.var target) (.bin "+" base (.litUint 1))) (.member m "_m01")
  , .assign (.index (.var target) (.bin "+" base (.litUint 2))) (.member m "_m02")
  , .assign (.index (.var target) (.bin "+" base (.litUint 3))) (.member m "_m10")
  , .assign (.index (.var target) (.bin "+" base (.litUint 4))) (.member m "_m11")
  , .assign (.index (.var target) (.bin "+" base (.litUint 5))) (.member m "_m12")
  , .assign (.index (.var target) (.bin "+" base (.litUint 6))) (.member m "_m20")
  , .assign (.index (.var target) (.bin "+" base (.litUint 7))) (.member m "_m21")
  , .assign (.index (.var target) (.bin "+" base (.litUint 8))) (.member m "_m22") ]

/-- Per-(row, col) write into the per-triangle row-major 3×3 matrix
    buffer named `target`, at base offset `base = 9 * tri`. -/
def setM (target : String) (base i j : SlangExpr) (v : SlangExpr) : SlangStmt :=
  .assign (.index (.var target)
            (.bin "+" base (.bin "+" (.bin "*" i (.litUint 3)) j)))
          v

/-- Inline 3×3 inverse via cofactor expansion with a zero-determinant
    guard (returns the zero matrix when |det| < 1e-30 to prevent NaN
    propagation through downstream solves). -/
def inv3x3 : SlangFunctionDecl :=
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

/-! ## Cotangent helper — used by every Laplacian assembler. -/

/-- `cot_at_apex(apex, o1, o2) = dot(a, b) / length(cross(a, b))`
    where `a = o1 - apex`, `b = o2 - apex`. Returns 0 on degenerate
    triangles (|a × b| < 1e-30) to avoid NaN. -/
def cot_at_apex : SlangFunctionDecl :=
  { attrs := []
  , retType := floatTy
  , name := "cot_at_apex"
  , params :=
      [ ⟨"apex", f3Ty, Semantic.none, none, none, .qIn⟩
      , ⟨"o1",   f3Ty, Semantic.none, none, none, .qIn⟩
      , ⟨"o2",   f3Ty, Semantic.none, none, none, .qIn⟩ ]
  , body :=
      [ .declInit f3Ty "a" (.bin "-" (.var "o1") (.var "apex"))
      , .declInit f3Ty "b" (.bin "-" (.var "o2") (.var "apex"))
      , .declInit floatTy "d"
          (.call "dot" [.var "a", .var "b"])
      , .declInit floatTy "cl"
          (.call "length" [.call "cross" [.var "a", .var "b"]])
      , .retExpr
          (.ternary (.bin "<" (.var "cl") (.litFloat 1e-30))
            (.litFloat 0.0)
            (.bin "/" (.var "d") (.var "cl"))) ] }

/-! ## Knuth/Dekker error-free transformations (df32 dot reduction). -/

/-- `two_sum(a, b) → (hi, lo)` — exact a + b = hi + lo. -/
def two_sum : SlangFunctionDecl :=
  { attrs := [], retType := .named "void", name := "two_sum"
  , params := [fIn "a", fIn "b", fOut "hi", fOut "lo"]
  , body :=
      [ .declInit floatTy "h" (.bin "+" (.var "a") (.var "b"))
      , .declInit floatTy "bb" (.bin "-" (.var "h") (.var "a"))
      , .declInit floatTy "ah" (.bin "-" (.var "h") (.var "bb"))
      , .declInit floatTy "lo_a" (.bin "-" (.var "a") (.var "ah"))
      , .declInit floatTy "lo_b" (.bin "-" (.var "b") (.var "bb"))
      , .assign (.var "hi") (.var "h")
      , .assign (.var "lo") (.bin "+" (.var "lo_a") (.var "lo_b"))
      , .ret none ] }

/-- `quick_two_sum(a, b) → (hi, lo)` — faster variant when |a| ≥ |b|. -/
def quick_two_sum : SlangFunctionDecl :=
  { attrs := [], retType := .named "void", name := "quick_two_sum"
  , params := [fIn "a", fIn "b", fOut "hi", fOut "lo"]
  , body :=
      [ .declInit floatTy "h" (.bin "+" (.var "a") (.var "b"))
      , .declInit floatTy "t" (.bin "-" (.var "h") (.var "a"))
      , .assign (.var "hi") (.var "h")
      , .assign (.var "lo") (.bin "-" (.var "b") (.var "t"))
      , .ret none ] }

/-- `two_prod(a, b) → (hi, lo)` — exact a · b = hi + lo via FMA. -/
def two_prod : SlangFunctionDecl :=
  { attrs := [], retType := .named "void", name := "two_prod"
  , params := [fIn "a", fIn "b", fOut "hi", fOut "lo"]
  , body :=
      [ .declInit floatTy "h" (.bin "*" (.var "a") (.var "b"))
      , .assign (.var "hi") (.var "h")
      , .assign (.var "lo")
          (.call "fma" [.var "a", .var "b", .un "-" (.var "h")])
      , .ret none ] }

/-- df + df → df (sloppy variant from Da Graça-Defour 2006 §3.2). -/
def df_add : SlangFunctionDecl :=
  { attrs := [], retType := .named "void", name := "df_add"
  , params :=
      [ fIn "x_hi", fIn "x_lo", fIn "y_hi", fIn "y_lo"
      , fOut "z_hi", fOut "z_lo" ]
  , body :=
      [ .declare floatTy "sh" none
      , .declare floatTy "sl" none
      , .expr (.call "two_sum"
          [.var "x_hi", .var "y_hi", .var "sh", .var "sl"])
      , .declInit floatTy "xy_lo"
          (.bin "+" (.var "x_lo") (.var "y_lo"))
      , .declInit floatTy "sl2"
          (.bin "+" (.var "sl") (.var "xy_lo"))
      , .expr (.call "quick_two_sum"
          [.var "sh", .var "sl2", .var "z_hi", .var "z_lo"])
      , .ret none ] }

/-- All four EFT helpers in dispatch order. Reach for this when a
    kernel needs the full df32 ladder (dot reduce, df-norm, etc.). -/
def eftHelpers : List SlangFunctionDecl :=
  [two_sum, quick_two_sum, two_prod, df_add]

end Curvenet.SlangCodegen.Common
