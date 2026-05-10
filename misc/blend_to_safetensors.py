"""Export Mire.blend (or any rigged Blender file) into a single
.safetensors holding every mesh's vertex attributes plus the armature
bone hierarchy + inverse-bind matrices.

Usage:

    blender --background <file.blend> \\
        --python misc/blend_to_safetensors.py -- <out.safetensors>

Or, when invoked from a Blender that already has the file open (e.g.
via Blender MCP), pass the output path as the first --argv:

    bpy.ops.script.python_file_run(filepath="misc/blend_to_safetensors.py")

The output schema is documented in `lean/SafetensorsToGltf.lean` —
this script is its inverse on the Blender side. No bmesh subdivision;
each loop_triangle vertex is emitted once (no fan dedup) so the
output exactly matches Blender's loop layout.
"""

import bpy
import json
import struct
import sys
from pathlib import Path


# ---------------------------------------------------------------- safetensors

def _safetensors_blob(entries, metadata):
    """Build a safetensors byte stream.

    entries: list of (name, dtype_str, shape_list, raw_bytes).
             Tensor offsets are assigned in input order.
    metadata: dict[str, str]. safetensors mandates string-only values.
    """
    header = {}
    if metadata:
        header["__metadata__"] = {k: str(v) for k, v in metadata.items()}
    body = bytearray()
    for name, dtype, shape, data in entries:
        lo = len(body)
        body += data
        hi = len(body)
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [lo, hi],
        }
    hdr_str = json.dumps(header, separators=(",", ":"))
    # Pad header so body starts on an 8-byte boundary (matches the
    # Python `safetensors` lib's behavior; lets mmap consumers align).
    prelude = 8 + len(hdr_str.encode("utf-8"))
    pad = (-prelude) % 8
    hdr_str = hdr_str + (" " * pad)
    hdr_bytes = hdr_str.encode("utf-8")
    out = struct.pack("<Q", len(hdr_bytes))
    out += hdr_bytes
    out += bytes(body)
    return out


# ---------------------------------------------------------------- mesh extract

def _safe_name(s: str) -> str:
    # safetensors tolerates most chars in tensor names, but '.' is the
    # group-separator we'll use, and spaces / brackets confuse JSON
    # tooling on the Lean side. Normalise.
    return (
        s.replace(".", "_")
         .replace(" ", "_")
         .replace("/", "_")
         .replace("[", "_")
         .replace("]", "_")
    )


def _per_vertex_top4(vert, vg_to_bone):
    """Collect the top-4 (joint_idx, weight) for one vertex, padded
    + normalised so weights sum to 1. Joints with no global mapping
    or zero weight are dropped."""
    pairs = []
    for g in vert.groups:
        bi = vg_to_bone[g.group] if g.group < len(vg_to_bone) else -1
        if bi < 0 or g.weight <= 0.0:
            continue
        pairs.append((bi, g.weight))
    pairs.sort(key=lambda jw: jw[1], reverse=True)
    pairs = pairs[:4]
    while len(pairs) < 4:
        pairs.append((0, 0.0))
    total = sum(w for _, w in pairs)
    if total > 0:
        pairs = [(j, w / total) for j, w in pairs]
    return pairs


def _extract_mesh(obj, vg_to_bone, uv_flip_v=True):
    """Return per-loop-triangle expanded attribute lists for one mesh.

    Returns a dict: positions/normals/uvs/joints/weights/indices.
    `indices` is the trivial 0..n_loop_verts-1 sequence (no fan dedup).
    """
    me = obj.data
    me.calc_loop_triangles()
    # Force per-vertex normal evaluation if the mesh has split (custom)
    # normals; otherwise v.normal is the mesh's auto-computed smooth
    # normal which is fine for our purposes.
    try:
        me.calc_normals_split()
    except (AttributeError, RuntimeError):
        # Newer Blender computes split normals lazily.
        pass

    has_uv = bool(me.uv_layers.active)
    uv_data = me.uv_layers.active.data if has_uv else None

    n_tris = len(me.loop_triangles)
    positions = []
    normals = []
    uvs = []
    joints = []
    weights = []
    for tri in me.loop_triangles:
        for k in range(3):
            vi = tri.vertices[k]
            li = tri.loops[k]
            v = me.vertices[vi]
            positions += [v.co.x, v.co.y, v.co.z]

            # Prefer split (per-loop) normal if the mesh provides one;
            # fall back to per-vertex normal.
            try:
                ln = me.loops[li].normal
                if ln.length_squared > 1e-12:
                    normals += [ln.x, ln.y, ln.z]
                else:
                    normals += [v.normal.x, v.normal.y, v.normal.z]
            except AttributeError:
                normals += [v.normal.x, v.normal.y, v.normal.z]

            if uv_data is not None:
                uv = uv_data[li].uv
                uvs += [uv.x, (1.0 - uv.y) if uv_flip_v else uv.y]
            else:
                uvs += [0.0, 0.0]

            top4 = _per_vertex_top4(v, vg_to_bone)
            for ji, wi in top4:
                joints.append(ji)
                weights.append(wi)

    n_verts = n_tris * 3
    indices = list(range(n_verts))
    return {
        "n_verts": n_verts,
        "n_tris": n_tris,
        "positions": positions,
        "normals": normals,
        "uvs": uvs,
        "joints": joints,
        "weights": weights,
        "indices": indices,
    }


# ---------------------------------------------------------------- main

def export_to_safetensors(out_path: Path) -> None:
    armature_obj = next((o for o in bpy.data.objects if o.type == "ARMATURE"), None)
    if armature_obj is None:
        raise SystemExit("blend_to_safetensors: no Armature found")

    bones = list(armature_obj.data.bones)
    bone_idx = {b.name: i for i, b in enumerate(bones)}
    n_bones = len(bones)

    bone_parents = [
        bone_idx.get(b.parent.name, -1) if b.parent else -1 for b in bones
    ]

    # Bone TRS in PARENT space (or armature space for roots). This is
    # what each bone's glTF Node carries as translation/rotation/scale.
    bone_translation = []
    bone_rotation = []  # XYZW
    bone_scale = []
    for b in bones:
        if b.parent:
            local = b.parent.matrix_local.inverted() @ b.matrix_local
        else:
            local = b.matrix_local
        loc, rot, sca = local.decompose()
        bone_translation += [loc.x, loc.y, loc.z]
        # mathutils.Quaternion is (W, X, Y, Z); glTF wants XYZW.
        bone_rotation += [rot.x, rot.y, rot.z, rot.w]
        bone_scale += [sca.x, sca.y, sca.z]

    # IBM = inverse(armature.matrix_world @ bone.matrix_local) — the
    # transform that takes a vertex from mesh-space (which equals
    # armature-world for skinned meshes) into bone-local at bind.
    # glTF stores mat4 as 16 floats column-major.
    Mw = armature_obj.matrix_world.copy()
    bone_ibm = []
    for b in bones:
        ibm = (Mw @ b.matrix_local).inverted()
        for col in range(4):
            for row in range(4):
                bone_ibm.append(ibm[row][col])

    # Walk meshes.
    entries = []
    mesh_names = []
    mesh_n_verts = []
    mesh_n_tris = []
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        # Skip meshes with no faces.
        if not obj.data.polygons:
            continue
        vg_to_bone = [bone_idx.get(vg.name, -1) for vg in obj.vertex_groups]
        m = _extract_mesh(obj, vg_to_bone)
        nm = _safe_name(obj.name)
        mesh_names.append(obj.name)
        mesh_n_verts.append(m["n_verts"])
        mesh_n_tris.append(m["n_tris"])

        nv = m["n_verts"]
        nt = m["n_tris"]
        entries.append((
            f"mesh.{nm}.positions", "F32", [nv, 3],
            struct.pack(f"<{nv*3}f", *m["positions"])
        ))
        entries.append((
            f"mesh.{nm}.normals", "F32", [nv, 3],
            struct.pack(f"<{nv*3}f", *m["normals"])
        ))
        entries.append((
            f"mesh.{nm}.uvs", "F32", [nv, 2],
            struct.pack(f"<{nv*2}f", *m["uvs"])
        ))
        entries.append((
            f"mesh.{nm}.joints", "U16", [nv, 4],
            struct.pack(f"<{nv*4}H", *m["joints"])
        ))
        entries.append((
            f"mesh.{nm}.weights", "F32", [nv, 4],
            struct.pack(f"<{nv*4}f", *m["weights"])
        ))
        entries.append((
            f"mesh.{nm}.indices", "U32", [nt, 3],
            struct.pack(f"<{nt*3}I", *m["indices"])
        ))

    # Bone tensors (single global skin palette).
    entries.append((
        "bone.ibm", "F32", [n_bones, 4, 4],
        struct.pack(f"<{n_bones*16}f", *bone_ibm)
    ))
    entries.append((
        "bone.translation", "F32", [n_bones, 3],
        struct.pack(f"<{n_bones*3}f", *bone_translation)
    ))
    entries.append((
        "bone.rotation", "F32", [n_bones, 4],
        struct.pack(f"<{n_bones*4}f", *bone_rotation)
    ))
    entries.append((
        "bone.scale", "F32", [n_bones, 3],
        struct.pack(f"<{n_bones*3}f", *bone_scale)
    ))

    metadata = {
        "format": "curvenet-mesh-v1",
        "source_axes": "Z-up",  # Blender / Mire native; consumer rotates via root node
        "armature_name": armature_obj.name,
        "bone_names": json.dumps([b.name for b in bones]),
        "bone_parents": json.dumps(bone_parents),
        "mesh_names": json.dumps(mesh_names),
        "mesh_n_verts": json.dumps(mesh_n_verts),
        "mesh_n_tris": json.dumps(mesh_n_tris),
    }

    blob = _safetensors_blob(entries, metadata)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(blob)
    print(
        f"wrote {out_path} ({out_path.stat().st_size:,} bytes; "
        f"{len(mesh_names)} meshes, {n_bones} bones, "
        f"{sum(mesh_n_verts):,} loop-verts, {sum(mesh_n_tris):,} tris)"
    )


def _argv_after_dashdash():
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1:]
    return []


if __name__ == "__main__":
    args = _argv_after_dashdash()
    out = Path(args[0]) if args else Path("/tmp/mire/mire.safetensors")
    export_to_safetensors(out)
