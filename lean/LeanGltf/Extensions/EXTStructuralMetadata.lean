import LeanGltf.Types

/-!
# `EXT_structural_metadata` — Cesium 3D Tiles Next vendor extension

Schema-driven structural metadata on glTF assets. Spec:
<https://github.com/CesiumGS/glTF/tree/3d-tiles-next/extensions/2.0/Vendor/EXT_structural_metadata>

We use this for the curvenet rig data (per-curve / per-segment /
per-knot tables), Direct Delta Mush weight harvest output, and any
other Curvenet-specific data that doesn't fit standard glTF
attributes. Extension name is registered as **optional** — readers
that don't understand it can still load the underlying mesh.

Surface in this module is writer-only: typed Lean structures plus
`toJson` builders that produce `LeanGltf.JSON.Value` ready to slot
into `Document.extensions` under the key `"EXT_structural_metadata"`.
-/

namespace LeanGltf.Extensions.EXTStructuralMetadata

open LeanGltf.JSON

/-! ## Class properties -/

/-- Property in a class definition. `componentType` is required when
    `type` is one of `SCALAR / VEC2 / VEC3 / VEC4 / MATN / ENUM`;
    omit otherwise. -/
structure ClassProperty where
  name          : Option String := none
  description   : Option String := none
  /-- One of `SCALAR`, `VEC2`..`VEC4`, `MAT2`..`MAT4`, `STRING`,
      `BOOLEAN`, `ENUM`. -/
  type          : String
  /-- One of `INT8`..`INT64`, `UINT8`..`UINT64`, `FLOAT32`,
      `FLOAT64` for numeric `type`s. -/
  componentType : Option String := none
  /-- Reference into `schema.enums` when `type = "ENUM"`. -/
  enumType      : Option String := none
  /-- Whether the property is an array (variable or fixed length). -/
  array         : Bool := false
  /-- Fixed array length (omit for variable length). -/
  count         : Option Nat := none
  /-- Whether the property is required on every entity. -/
  required      : Bool := false

namespace ClassProperty

def toJson (p : ClassProperty) : Value :=
  obj? #[
    ("name",          p.name.map .str),
    ("description",   p.description.map .str),
    ("type",          some (.str p.type)),
    ("componentType", p.componentType.map .str),
    ("enumType",      p.enumType.map .str),
    ("array",         if p.array then some (.bool true) else none),
    ("count",         p.count.map (fun n => .int (Int.ofNat n))),
    ("required",      if p.required then some (.bool true) else none)
  ]

end ClassProperty

/-! ## Class definition -/

structure ClassDef where
  name        : Option String := none
  description : Option String := none
  /-- Property name → property definition. -/
  properties  : Array (String × ClassProperty)

namespace ClassDef

def toJson (c : ClassDef) : Value :=
  let propsObj : Value :=
    .obj (c.properties.map (fun (k, p) => (k, ClassProperty.toJson p)))
  obj? #[
    ("name",        c.name.map .str),
    ("description", c.description.map .str),
    ("properties",  some propsObj)
  ]

end ClassDef

/-! ## Schema -/

structure Schema where
  id          : Option String := none
  name        : Option String := none
  description : Option String := none
  version     : Option String := none
  /-- Class name → class definition. -/
  classes     : Array (String × ClassDef) := #[]

namespace Schema

def toJson (s : Schema) : Value :=
  let classesObj : Option Value :=
    if s.classes.isEmpty then none
    else some (.obj (s.classes.map (fun (k, c) => (k, ClassDef.toJson c))))
  obj? #[
    ("id",          s.id.map .str),
    ("name",        s.name.map .str),
    ("description", s.description.map .str),
    ("version",     s.version.map .str),
    ("classes",     classesObj)
  ]

end Schema

/-! ## Property table — column-based storage of class instances -/

/-- One column inside a property table. `values` is a buffer-view
    index (the actual bytes live in the GLB BIN chunk). For variable-
    length arrays of strings, `arrayOffsets` and `stringOffsets`
    point at additional buffer views; we don't model those for the
    MVP since the curvenet data is fixed-length numeric. -/
structure PropertyTableProperty where
  values        : Nat                       -- bufferView index
  arrayOffsets  : Option Nat := none
  stringOffsets : Option Nat := none

namespace PropertyTableProperty

def toJson (p : PropertyTableProperty) : Value :=
  obj? #[
    ("values",        some (.int (Int.ofNat p.values))),
    ("arrayOffsets",  p.arrayOffsets.map (fun n => .int (Int.ofNat n))),
    ("stringOffsets", p.stringOffsets.map (fun n => .int (Int.ofNat n)))
  ]

end PropertyTableProperty

structure PropertyTable where
  name       : Option String := none
  /-- Class id this table is an instance set of. -/
  classId    : String
  /-- Number of instances (rows) in the table. -/
  count      : Nat
  /-- Property name → column. -/
  properties : Array (String × PropertyTableProperty) := #[]

namespace PropertyTable

def toJson (t : PropertyTable) : Value :=
  let propsObj : Option Value :=
    if t.properties.isEmpty then none
    else some (.obj (t.properties.map
      (fun (k, p) => (k, PropertyTableProperty.toJson p))))
  obj? #[
    ("name",       t.name.map .str),
    ("class",      some (.str t.classId)),
    ("count",      some (.int (Int.ofNat t.count))),
    ("properties", propsObj)
  ]

end PropertyTable

/-! ## Property attribute — per-vertex structural metadata -/

/-- One per-vertex attribute mapped onto a glTF attribute name.
    The `attribute` string is the primitive's attribute key (e.g.
    `"_DDM_HANDLES"`, `"_CURVENET_SAMPLE_INDEX"`). Underscore-
    prefixed names are the glTF convention for application-specific
    attributes. -/
structure PropertyAttributeProperty where
  attributeName : String

namespace PropertyAttributeProperty

def toJson (p : PropertyAttributeProperty) : Value :=
  obj? #[
    ("attribute", some (.str p.attributeName))
  ]

end PropertyAttributeProperty

structure PropertyAttribute where
  name       : Option String := none
  classId    : String
  properties : Array (String × PropertyAttributeProperty) := #[]

namespace PropertyAttribute

def toJson (a : PropertyAttribute) : Value :=
  let propsObj : Option Value :=
    if a.properties.isEmpty then none
    else some (.obj (a.properties.map
      (fun (k, p) => (k, PropertyAttributeProperty.toJson p))))
  obj? #[
    ("name",       a.name.map .str),
    ("class",      some (.str a.classId)),
    ("properties", propsObj)
  ]

end PropertyAttribute

/-! ## The full top-level extension object -/

structure Top where
  schema             : Option Schema := none
  schemaUri          : Option String := none
  propertyTables     : Array PropertyTable := #[]
  propertyAttributes : Array PropertyAttribute := #[]

namespace Top

/-- The extension key under which this object is keyed in
    `Document.extensions`. -/
def extensionName : String := "EXT_structural_metadata"

def toJson (t : Top) : Value :=
  obj? #[
    ("schema",             t.schema.map Schema.toJson),
    ("schemaUri",          t.schemaUri.map .str),
    ("propertyTables",     if t.propertyTables.isEmpty     then none
                           else some (.arr (t.propertyTables.map     PropertyTable.toJson))),
    ("propertyAttributes", if t.propertyAttributes.isEmpty then none
                           else some (.arr (t.propertyAttributes.map PropertyAttribute.toJson)))
  ]

end Top

end LeanGltf.Extensions.EXTStructuralMetadata
