import LeanGltf.JSON

/-!
# `LeanGltf` data types

A writer-only port of the glTF 2.0 graph from `aria-gltf` (Elixir, MIT,
K. S. Ernest "iFire" Lee — see `references.bib`).

Scope (mirrors the plan's "out of scope" list):
* No textures, images, samplers, or cameras.
* One default PBR material.
* No morph targets, no extensions, no `extras`.

The structures intentionally use `Option` for every non-required field so
the JSON encoder can omit them, matching the glTF reference behaviour.
-/

namespace LeanGltf
open LeanGltf.JSON

/-! ## Asset (`asset.ex`) -/

structure Asset where
  version   : String := "2.0"
  generator : Option String := some "AC11Lean / LeanGltf"
  copyright : Option String := none
  deriving Repr

namespace Asset
  def toJson (a : Asset) : Value :=
    obj? #[
      ("version",   some (.str a.version)),
      ("generator", a.generator.map .str),
      ("copyright", a.copyright.map .str)
    ]
end Asset

/-! ## Buffer (`buffer.ex`) -/

structure Buffer where
  byteLength : Nat
  uri        : Option String := none      -- nil ⇒ embedded GLB chunk
  deriving Repr

namespace Buffer
  def toJson (b : Buffer) : Value :=
    obj? #[
      ("byteLength", some (.int (Int.ofNat b.byteLength))),
      ("uri",        b.uri.map .str)
    ]
end Buffer

/-! ## BufferView (`buffer_view.ex`) -/

structure BufferView where
  buffer     : Nat                -- index into Document.buffers
  byteOffset : Nat := 0
  byteLength : Nat
  byteStride : Option Nat := none
  /-- glTF target hint: 34962 = ARRAY_BUFFER, 34963 = ELEMENT_ARRAY_BUFFER. -/
  target     : Option Nat := none
  deriving Repr

namespace BufferView
  def arrayBuffer        : Nat := 34962
  def elementArrayBuffer : Nat := 34963

  def toJson (v : BufferView) : Value :=
    obj? #[
      ("buffer",     some (.int (Int.ofNat v.buffer))),
      ("byteOffset", if v.byteOffset = 0 then none else some (.int (Int.ofNat v.byteOffset))),
      ("byteLength", some (.int (Int.ofNat v.byteLength))),
      ("byteStride", v.byteStride.map (fun n => .int (Int.ofNat n))),
      ("target",     v.target.map     (fun n => .int (Int.ofNat n)))
    ]
end BufferView

/-! ## Accessor (`accessor.ex`)

Component types follow the WebGL constants used in glTF:
  5120 BYTE / 5121 UBYTE / 5122 SHORT / 5123 USHORT / 5125 UINT / 5126 FLOAT.
-/

inductive AccessorType
  | scalar | vec2 | vec3 | vec4 | mat2 | mat3 | mat4
  deriving Repr, DecidableEq, Inhabited

namespace AccessorType
  def toString : AccessorType → String
    | .scalar => "SCALAR" | .vec2 => "VEC2" | .vec3 => "VEC3" | .vec4 => "VEC4"
    | .mat2   => "MAT2"   | .mat3 => "MAT3" | .mat4 => "MAT4"

  def componentCount : AccessorType → Nat
    | .scalar => 1 | .vec2 => 2 | .vec3 => 3 | .vec4 => 4
    | .mat2   => 4 | .mat3 => 9 | .mat4 => 16
end AccessorType

structure Accessor where
  bufferView    : Option Nat := none
  byteOffset    : Nat := 0
  componentType : Nat                -- 5120..5126
  count         : Nat
  type          : AccessorType
  /-- Componentwise min/max — required by glTF for POSITION accessors,
  optional elsewhere. We always emit them when set. -/
  min           : Option (Array Float) := none
  max           : Option (Array Float) := none
  normalized    : Bool := false
  deriving Repr

namespace Accessor
  def byteSize : Nat → Nat
    | 5120 => 1 | 5121 => 1 | 5122 => 2 | 5123 => 2 | 5125 => 4 | 5126 => 4
    | _    => 0

  def elementByteSize (a : Accessor) : Nat :=
    byteSize a.componentType * a.type.componentCount

  def toJson (a : Accessor) : Value :=
    obj? #[
      ("bufferView",    a.bufferView.map (fun n => .int (Int.ofNat n))),
      ("byteOffset",    if a.byteOffset = 0 then none else some (.int (Int.ofNat a.byteOffset))),
      ("componentType", some (.int (Int.ofNat a.componentType))),
      ("count",         some (.int (Int.ofNat a.count))),
      ("type",          some (.str a.type.toString)),
      ("min",           a.min.map JSON.ofFloatArr),
      ("max",           a.max.map JSON.ofFloatArr),
      ("normalized",    if a.normalized then some (.bool true) else none)
    ]
end Accessor

/-! ## Mesh / Primitive (`mesh.ex` + `mesh/primitive.ex`) -/

/-- glTF primitive topology. We only emit `triangles = 4`. -/
def TRIANGLES : Nat := 4

structure Primitive where
  /-- Attribute name → accessor index, e.g. `("POSITION", 0)`, `("NORMAL", 1)`,
  `("JOINTS_0", 2)`, `("WEIGHTS_0", 3)`, `("TEXCOORD_0", 4)`. -/
  attributes : Array (String × Nat)
  indices    : Option Nat := none
  material   : Option Nat := none
  mode       : Nat := TRIANGLES
  /-- Morph targets, per glTF 2.0 §3.6.2.1. Each target is its own
  attribute-name → accessor-index map (POSITION / NORMAL / TANGENT
  *deltas* relative to the base attribute). The host mesh's `weights`
  picks the blend per target. -/
  targets    : Array (Array (String × Nat)) := #[]
  deriving Repr

namespace Primitive
  def toJson (p : Primitive) : Value :=
    let attrs : Value :=
      .obj (p.attributes.map (fun (k, n) => (k, .int (Int.ofNat n))))
    let targetsJson : Option Value :=
      if p.targets.isEmpty then none
      else some (.arr (p.targets.map (fun t =>
        .obj (t.map (fun (k, n) => (k, .int (Int.ofNat n)))))))
    obj? #[
      ("attributes", some attrs),
      ("indices",    p.indices.map (fun n => .int (Int.ofNat n))),
      ("material",   p.material.map (fun n => .int (Int.ofNat n))),
      ("mode",       if p.mode = TRIANGLES then none else some (.int (Int.ofNat p.mode))),
      ("targets",    targetsJson)
    ]
end Primitive

structure Mesh where
  primitives : Array Primitive
  name       : Option String := none
  /-- Initial morph-target weights, one entry per target on every
  primitive (glTF requires all primitives in a mesh share target count).
  Animations bind to this via `AnimPath.weights`. -/
  weights    : Array Float := #[]
  deriving Repr

namespace Mesh
  def toJson (m : Mesh) : Value :=
    obj? #[
      ("primitives", some (.arr (m.primitives.map Primitive.toJson))),
      ("name",       m.name.map .str),
      ("weights",    if m.weights.isEmpty then none else some (JSON.ofFloatArr m.weights))
    ]
end Mesh

/-! ## Skin (`skin.ex`) -/

structure Skin where
  /-- Node indices that form the joint hierarchy. Order = palette index. -/
  joints              : Array Nat
  /-- Accessor index of an array of `mat4` inverse-bind matrices (count
  must match `joints.size`). Required for skinning to work in importers. -/
  inverseBindMatrices : Option Nat := none
  /-- Optional: node index used as the skeleton root. When set, must be
  a member of `joints`. -/
  skeleton            : Option Nat := none
  name                : Option String := none
  deriving Repr

namespace Skin
  def toJson (s : Skin) : Value :=
    obj? #[
      ("joints",              some (JSON.ofIntArr s.joints)),
      ("inverseBindMatrices", s.inverseBindMatrices.map (fun n => .int (Int.ofNat n))),
      ("skeleton",            s.skeleton.map            (fun n => .int (Int.ofNat n))),
      ("name",                s.name.map .str)
    ]
end Skin

/-! ## Node (`node.ex`) — TRS-only (no matrix) -/

structure Node where
  name        : Option String := none
  /-- Translation, rotation (quaternion XYZW), scale. Default is identity. -/
  translation : Option (Array Float) := none    -- 3 floats
  rotation    : Option (Array Float) := none    -- 4 floats, XYZW
  scale       : Option (Array Float) := none    -- 3 floats
  children    : Array Nat := #[]
  mesh        : Option Nat := none
  skin        : Option Nat := none
  deriving Repr

namespace Node
  def toJson (n : Node) : Value :=
    let kids := if n.children.isEmpty then none else some (JSON.ofIntArr n.children)
    obj? #[
      ("name",        n.name.map .str),
      ("translation", n.translation.map JSON.ofFloatArr),
      ("rotation",    n.rotation.map    JSON.ofFloatArr),
      ("scale",       n.scale.map       JSON.ofFloatArr),
      ("children",    kids),
      ("mesh",        n.mesh.map (fun i => .int (Int.ofNat i))),
      ("skin",        n.skin.map (fun i => .int (Int.ofNat i)))
    ]
end Node

/-! ## Animation (`animation.ex` + channel + sampler) -/

inductive Interpolation
  | linear | step | cubicspline
  deriving Repr, DecidableEq

namespace Interpolation
  def toString : Interpolation → String
    | .linear => "LINEAR" | .step => "STEP" | .cubicspline => "CUBICSPLINE"
end Interpolation

inductive AnimPath
  | translation | rotation | scale | weights
  deriving Repr, DecidableEq

namespace AnimPath
  def toString : AnimPath → String
    | .translation => "translation" | .rotation => "rotation"
    | .scale       => "scale"       | .weights  => "weights"
end AnimPath

structure AnimSampler where
  input         : Nat            -- accessor index (times)
  output        : Nat            -- accessor index (values)
  interpolation : Interpolation := .linear
  deriving Repr

namespace AnimSampler
  def toJson (s : AnimSampler) : Value :=
    obj? #[
      ("input",         some (.int (Int.ofNat s.input))),
      ("output",        some (.int (Int.ofNat s.output))),
      -- LINEAR is the default, omit for compactness.
      ("interpolation",
        if s.interpolation = .linear then none
        else some (.str s.interpolation.toString))
    ]
end AnimSampler

structure AnimTarget where
  node : Option Nat := none
  path : AnimPath
  deriving Repr

namespace AnimTarget
  def toJson (t : AnimTarget) : Value :=
    obj? #[
      ("node", t.node.map (fun i => .int (Int.ofNat i))),
      ("path", some (.str t.path.toString))
    ]
end AnimTarget

structure AnimChannel where
  sampler : Nat                 -- index into Animation.samplers
  target  : AnimTarget
  deriving Repr

namespace AnimChannel
  def toJson (c : AnimChannel) : Value :=
    obj? #[
      ("sampler", some (.int (Int.ofNat c.sampler))),
      ("target",  some (AnimTarget.toJson c.target))
    ]
end AnimChannel

structure Animation where
  channels : Array AnimChannel
  samplers : Array AnimSampler
  name     : Option String := none
  deriving Repr

namespace Animation
  def toJson (a : Animation) : Value :=
    obj? #[
      ("channels", some (.arr (a.channels.map AnimChannel.toJson))),
      ("samplers", some (.arr (a.samplers.map AnimSampler.toJson))),
      ("name",     a.name.map .str)
    ]
end Animation

/-! ## Scene (`scene.ex`) -/

structure Scene where
  nodes : Array Nat := #[]
  name  : Option String := none
  deriving Repr

namespace Scene
  def toJson (s : Scene) : Value :=
    obj? #[
      ("nodes", if s.nodes.isEmpty then none else some (JSON.ofIntArr s.nodes)),
      ("name",  s.name.map .str)
    ]
end Scene

/-! ## Material (default PBR fallback)

We don't expose materials as a real port — every primitive shares one
default PBR material. -/

structure Material where
  name : Option String := some "default"
  deriving Repr

namespace Material
  /-- A bland grey PBR. Importers will accept this and produce a default
  shader. -/
  def toJson (m : Material) : Value :=
    obj? #[
      ("name", m.name.map .str),
      ("pbrMetallicRoughness", some (.obj #[
          ("baseColorFactor", JSON.ofFloatArr #[0.8, 0.8, 0.8, 1.0]),
          ("metallicFactor",  .num 0.0),
          ("roughnessFactor", .num 0.9)
        ]))
    ]
end Material

end LeanGltf
