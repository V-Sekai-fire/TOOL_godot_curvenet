/-!
# Minimal JSON encoder for `LeanGltf`

A pragmatic JSON value type sufficient for emitting glTF 2.0 documents.
We avoid `Lean.Data.Json` (which would pull in the Lean elaborator) and
hand-roll a tiny encoder so we control float formatting precisely — the
glTF spec is permissive but importers (e.g. Blender's `io_scene_gltf2`)
can be picky about NaN/infinity and trailing zeros.

Adapted from the encoder shape used by `aria-gltf` (`AriaGltf.Document.to_json`,
MIT, K. S. Ernest "iFire" Lee — see `references.bib`).
-/

namespace LeanGltf.JSON

inductive Value where
  | null
  | bool   (b : Bool)
  | int    (n : Int)
  | num    (f : Float)        -- emitted with full precision
  | str    (s : String)
  | arr    (xs : Array Value)
  | obj    (kvs : Array (String × Value))
  deriving Inhabited

/-- Construct a JSON object from a list of (key, value) pairs, dropping
entries whose value is `none`. Mirrors aria-gltf's `put_if_present`. -/
def obj? (entries : Array (String × Option Value)) : Value :=
  Value.obj <| entries.filterMap (fun (k, v?) => v?.map (fun v => (k, v)))

/-! ## Helpers for the common option-emit pattern -/

def some? (v : Value) : Option Value := some v
def whenSome {α} (x : Option α) (f : α → Value) : Option Value := x.map f

/-! ## Float → string

Lean's `Float.toString` emits `0.000000`-style. That's valid JSON but
verbose. We use a slightly tighter form: integer-valued floats become
`"123.0"`, others fall through to `Float.toString`. NaN/inf are
serialised as `null` (per glTF spec — these aren't valid floats). -/

def floatToJsonString (f : Float) : String :=
  if f.isNaN || !f.isFinite then "null"
  else if f == f.floor && f.abs < 1e16 then
    -- Integer-valued, small enough to stringify exactly.
    let i : Int := if f < 0 then -((-f).toUInt64.toNat : Int) else (f.toUInt64.toNat : Int)
    s!"{i}.0"
  else
    f.toString

/-! ## String escaping (RFC 8259 minimum) -/

private def hex4 (n : Nat) : String :=
  let d := "0123456789abcdef".toList.toArray
  let nibble (k : Nat) : Char := d[(n >>> k) &&& 0xF]!
  String.ofList [nibble 12, nibble 8, nibble 4, nibble 0]

def escapeString (s : String) : String := Id.run do
  let mut out := "\""
  for c in s.toList do
    match c with
    | '\"' => out := out ++ "\\\""
    | '\\' => out := out ++ "\\\\"
    | '\n' => out := out ++ "\\n"
    | '\r' => out := out ++ "\\r"
    | '\t' => out := out ++ "\\t"
    | '\x08' => out := out ++ "\\b"
    | '\x0c' => out := out ++ "\\f"
    | c =>
      let n := c.toNat
      if n < 0x20 then out := out ++ "\\u" ++ hex4 n
      else out := out.push c
  out := out ++ "\""
  pure out

/-! ## Render -/

partial def render : Value → String
  | .null      => "null"
  | .bool true => "true"
  | .bool false => "false"
  | .int n     => toString n
  | .num f     => floatToJsonString f
  | .str s     => escapeString s
  | .arr xs    =>
    let parts := xs.map render
    "[" ++ String.intercalate "," parts.toList ++ "]"
  | .obj kvs   =>
    let parts := kvs.map (fun (k, v) => escapeString k ++ ":" ++ render v)
    "{" ++ String.intercalate "," parts.toList ++ "}"

/-! ## Convenience builders -/

@[inline] def ofInt   (n : Int)    : Value := .int n
@[inline] def ofNat   (n : Nat)    : Value := .int (Int.ofNat n)
@[inline] def ofFloat (f : Float)  : Value := .num f
@[inline] def ofStr   (s : String) : Value := .str s
@[inline] def ofBool  (b : Bool)   : Value := .bool b

def ofIntArr (xs : Array Nat) : Value :=
  .arr (xs.map (fun n => .int (Int.ofNat n)))

def ofFloatArr (xs : Array Float) : Value :=
  .arr (xs.map (fun f => .num f))

end LeanGltf.JSON
