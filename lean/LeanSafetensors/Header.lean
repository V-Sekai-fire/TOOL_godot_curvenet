import LeanGltf.JSON

/-!
# safetensors header — typed JSON layer

A safetensors file is a fixed three-part container:

```
+0x00 u64 LE       header_len            -- bytes of JSON that follow
+0x08 …            UTF-8 JSON header     -- metadata + per-tensor info
+...  raw bytes    contiguous tensor data, offsets relative to here
```

The header JSON has shape:

```json
{
  "__metadata__": { "free": "form", "value": "is always a string" },
  "tensor_name":  { "dtype": "F32", "shape": [n,3], "data_offsets": [a,b] },
  ...
}
```

Spec: <https://github.com/huggingface/safetensors>.

This module is the typed `Header` ↔ JSON layer. `Reader` and `Writer`
sit on top.
-/

namespace LeanSafetensors

/-- safetensors dtype tags. We carry the full set the spec defines —
    even the ones the curvenet pipeline doesn't currently use — so
    third-party files round-trip without information loss. -/
inductive Dtype where
  | f32 | f64 | f16 | bf16
  | u8  | u16 | u32 | u64
  | i8  | i16 | i32 | i64
  | bool
  deriving Repr, DecidableEq, Inhabited

namespace Dtype

def toString : Dtype → String
  | .f32  => "F32"  | .f64  => "F64"  | .f16  => "F16"  | .bf16 => "BF16"
  | .u8   => "U8"   | .u16  => "U16"  | .u32  => "U32"  | .u64  => "U64"
  | .i8   => "I8"   | .i16  => "I16"  | .i32  => "I32"  | .i64  => "I64"
  | .bool => "BOOL"

def fromString : String → Except String Dtype
  | "F32"  => .ok .f32  | "F64"  => .ok .f64  | "F16"  => .ok .f16  | "BF16" => .ok .bf16
  | "U8"   => .ok .u8   | "U16"  => .ok .u16  | "U32"  => .ok .u32  | "U64"  => .ok .u64
  | "I8"   => .ok .i8   | "I16"  => .ok .i16  | "I32"  => .ok .i32  | "I64"  => .ok .i64
  | "BOOL" => .ok .bool
  | s      => .error s!"safetensors: unknown dtype \"{s}\""

def byteSize : Dtype → Nat
  | .f32 | .u32 | .i32          => 4
  | .f64 | .u64 | .i64          => 8
  | .f16 | .bf16 | .u16 | .i16  => 2
  | .u8  | .i8  | .bool         => 1

end Dtype

/-- Per-tensor record in the JSON header. Offsets are relative to the
    body section (i.e. start at 0 for the first tensor), not the file. -/
structure TensorInfo where
  dtype    : Dtype
  shape    : Array Nat
  offsetLo : Nat
  offsetHi : Nat
  deriving Repr, Inhabited

namespace TensorInfo

def numElements (t : TensorInfo) : Nat :=
  t.shape.foldl (· * ·) 1

def expectedByteSize (t : TensorInfo) : Nat :=
  t.dtype.byteSize * t.numElements

def actualByteSize (t : TensorInfo) : Nat :=
  t.offsetHi - t.offsetLo

end TensorInfo

/-- The deserialised header. Tensor order matches file order so the
    Writer can produce byte-stable round-trips when given the original
    `Header` back. -/
structure Header where
  tensors  : Array (String × TensorInfo) := #[]
  /-- Free-form metadata. safetensors mandates string-only values; if
      you want structured data here, JSON-encode it into the string. -/
  metadata : Array (String × String) := #[]
  deriving Repr, Inhabited

namespace Header

open LeanGltf.JSON

/-- Encode the header as a `JSON.Value`. `__metadata__` is emitted
    first when present, then tensor entries in declared order. -/
def toJsonValue (h : Header) : Value :=
  let tensorEntry (name : String) (t : TensorInfo) : (String × Value) :=
    let shapeArr : Value := .arr (t.shape.map (fun n => .int (Int.ofNat n)))
    let offsArr  : Value := .arr #[
      .int (Int.ofNat t.offsetLo),
      .int (Int.ofNat t.offsetHi)
    ]
    (name, .obj #[
      ("dtype",        .str t.dtype.toString),
      ("shape",        shapeArr),
      ("data_offsets", offsArr)
    ])
  let metaPair : Array (String × Value) :=
    if h.metadata.isEmpty then #[]
    else
      let mdEntries : Array (String × Value) :=
        h.metadata.map (fun (k, v) => (k, .str v))
      #[("__metadata__", .obj mdEntries)]
  let tensorPairs : Array (String × Value) :=
    h.tensors.map (fun (name, info) => tensorEntry name info)
  .obj (metaPair ++ tensorPairs)

def toJsonString (h : Header) : String := render (toJsonValue h)

private def lookupKV (kvs : Array (String × Value)) (key : String) : Option Value :=
  kvs.findSome? (fun (k, v) => if k = key then some v else none)

private def decodeTensorInfo (v : Value) : Except String TensorInfo := do
  match v with
  | .obj kvs =>
    let dtypeStr ← match lookupKV kvs "dtype" with
      | some (.str s) => pure s
      | _ => throw "safetensors: tensor entry missing/invalid dtype"
    let dtype ← Dtype.fromString dtypeStr
    let shape : Array Nat ← match lookupKV kvs "shape" with
      | some (.arr xs) =>
        xs.foldlM (init := (#[] : Array Nat)) (fun acc x =>
          match x with
          | .int n =>
            if n < 0 then throw "safetensors: negative shape entry"
            else pure (acc.push n.toNat)
          | _ => throw "safetensors: non-integer shape entry")
      | _ => throw "safetensors: tensor entry missing/invalid shape"
    let (lo, hi) ← match lookupKV kvs "data_offsets" with
      | some (.arr xs) =>
        if xs.size ≠ 2 then throw "safetensors: data_offsets must have 2 entries"
        else match xs[0]!, xs[1]! with
          | .int a, .int b =>
            if a < 0 || b < 0 then throw "safetensors: negative data_offsets entry"
            else pure (a.toNat, b.toNat)
          | _, _ => throw "safetensors: non-integer data_offsets entry"
      | _ => throw "safetensors: tensor entry missing/invalid data_offsets"
    pure { dtype := dtype, shape := shape, offsetLo := lo, offsetHi := hi }
  | _ => throw "safetensors: tensor entry is not an object"

/-- Parse a `JSON.Value` (the decoded header) into a `Header`. -/
def fromJson (v : Value) : Except String Header := do
  match v with
  | .obj kvs =>
    let mut metadata : Array (String × String) := #[]
    let mut tensors : Array (String × TensorInfo) := #[]
    for kv in kvs do
      let k  := kv.1
      let vv := kv.2
      if k = "__metadata__" then
        match vv with
        | .obj mds =>
          for mkv in mds do
            let mk := mkv.1
            let mv := mkv.2
            match mv with
            | .str s => metadata := metadata.push (mk, s)
            | _ =>
              -- safetensors mandates string-only values; coerce any
              -- non-string by re-rendering it back to JSON text. This
              -- is forgiving for files produced by tools that bend
              -- the rules.
              metadata := metadata.push (mk, render mv)
        | _ => throw "safetensors: __metadata__ is not an object"
      else
        let info ← decodeTensorInfo vv
        tensors := tensors.push (k, info)
    pure { tensors := tensors, metadata := metadata }
  | _ => throw "safetensors: header root is not an object"

end Header
end LeanSafetensors
