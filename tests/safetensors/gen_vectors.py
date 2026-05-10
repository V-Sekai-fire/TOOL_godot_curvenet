"""Generate a small but spec-complete set of safetensors test vectors.

Each vector exercises a distinct corner of the format. The Lean
LeanSafetensors implementation is then exercised against this set.

Coverage matrix:

    | vector        | dtypes              | shapes                        | metadata    | tensor count |
    |---------------|---------------------|-------------------------------|-------------|--------------|
    | empty         | —                   | —                             | none        | 0            |
    | metadata_only | —                   | —                             | yes         | 0            |
    | scalar_f32    | F32                 | []                            | none        | 1            |
    | small_f32     | F32                 | [3], [2,2]                    | yes         | 2            |
    | int_grid      | I8 I16 I32 I64      | [4], [2,2], [2,2,2], [3,3]    | yes         | 4            |
    | uint_grid     | U8 U16 U32 U64      | [4], [2,2], [2,2,2], [3,3]    | yes         | 4            |
    | floats        | F16 BF16 F32 F64    | [4], [4], [4], [4]            | yes         | 4            |
    | bool_v        | BOOL                | [8]                           | none        | 1            |
    | mesh_like     | F32 U32 U16         | [N,3], [M,3], [N,4]           | yes         | 3            |

The output files are committed-friendly (small) and reproducible —
all values are deterministic, no rng. Run from anywhere:

    python3 tests/safetensors/gen_vectors.py [out_dir]

Default out_dir is `tests/safetensors/vectors/`.
"""

import json
import struct
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file


def _dump(path: Path, tensors: dict[str, np.ndarray], metadata: dict | None):
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path), metadata=metadata)


def gen(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. empty — 0 tensors, no metadata
    _dump(out_dir / "empty.safetensors", {}, None)

    # 2. metadata_only — 0 tensors, populated metadata
    _dump(out_dir / "metadata_only.safetensors", {},
          {"format": "test-v1", "author": "curvenet", "shape_hint": json.dumps([1, 2, 3])})

    # 3. scalar_f32 — single 0-D tensor
    _dump(out_dir / "scalar_f32.safetensors",
          {"x": np.array(1.5, dtype=np.float32)},
          None)

    # 4. small_f32 — 1-D + 2-D, with metadata
    _dump(out_dir / "small_f32.safetensors",
          {
              "vec": np.array([1.0, 2.0, 3.0], dtype=np.float32),
              "mat": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
          },
          {"format": "test-v1"})

    # 5. int_grid — every signed int dtype
    _dump(out_dir / "int_grid.safetensors",
          {
              "i8":  np.array([-128, -1, 0, 127], dtype=np.int8),
              "i16": np.array([[-32768, -1], [0, 32767]], dtype=np.int16),
              "i32": np.arange(-4, 4, dtype=np.int32).reshape(2, 2, 2),
              "i64": np.arange(9, dtype=np.int64).reshape(3, 3),
          },
          {"format": "int-grid"})

    # 6. uint_grid — every unsigned int dtype
    _dump(out_dir / "uint_grid.safetensors",
          {
              "u8":  np.array([0, 1, 127, 255], dtype=np.uint8),
              "u16": np.array([[0, 1], [2, 65535]], dtype=np.uint16),
              "u32": np.arange(8, dtype=np.uint32).reshape(2, 2, 2),
              "u64": np.arange(9, dtype=np.uint64).reshape(3, 3),
          },
          {"format": "uint-grid"})

    # 7. floats — every float dtype
    _dump(out_dir / "floats.safetensors",
          {
              "f16":  np.array([0.0, 0.5, 1.0, -1.0], dtype=np.float16),
              "bf16": np.array([0.0, 0.5, 1.0, -1.0]).astype(np.float32).view(np.uint16)[1::2].astype(np.float32),
              "f32":  np.array([0.0, 0.5, 1.0, -1.0], dtype=np.float32),
              "f64":  np.array([0.0, 0.5, 1.0, -1.0], dtype=np.float64),
          },
          {"format": "floats"})
    # The bf16 trick above produces f32 not bf16 — Numpy doesn't ship bf16.
    # Fix by switching that file to use raw bytes via low-level write.
    _bf16_fix(out_dir / "floats.safetensors")

    # 8. bool_v — packs 1 byte per bool (safetensors convention)
    _dump(out_dir / "bool_v.safetensors",
          {"flags": np.array([True, False, True, False, True, False, True, False])},
          None)

    # 9. mesh_like — the curvenet shape: positions / indices / weights
    _dump(out_dir / "mesh_like.safetensors",
          {
              "positions": np.array(
                  [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                  dtype=np.float32),
              "indices":   np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32),
              "weights":   np.array(
                  [[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0],
                   [0.25, 0.25, 0.25, 0.25], [0.7, 0.2, 0.1, 0.0]],
                  dtype=np.float32),
          },
          {"format": "mesh-v1", "n_verts": "4", "n_tris": "2"})


def _bf16_fix(path: Path) -> None:
    """Replace the f32-masquerading-as-bf16 tensor in `floats.safetensors`
    with a real BF16 payload (4 elements: 0.0, 0.5, 1.0, -1.0).

    bf16 = top 16 bits of fp32. We bit-bash directly so we don't depend
    on numpy bf16 (which it lacks)."""
    fp32_bits = struct.pack("<4f", 0.0, 0.5, 1.0, -1.0)  # 16 bytes
    bf16_bytes = b"".join(fp32_bits[4 * i + 2 : 4 * i + 4] for i in range(4))  # 8 bytes
    assert len(bf16_bytes) == 8

    # Read the existing file, swap the bf16 tensor's bytes + dtype tag.
    raw = path.read_bytes()
    hlen = struct.unpack_from("<Q", raw, 0)[0]
    header = json.loads(raw[8 : 8 + hlen])
    assert "bf16" in header
    info = header["bf16"]
    assert info["shape"] == [4]
    lo, hi = info["data_offsets"]
    assert hi - lo == 16  # was f32: 4 elements × 4 bytes
    # Rewrite tensor data: replace the 16-byte f32 slice with the 8-byte bf16 payload.
    body_start = 8 + hlen
    body = bytearray(raw[body_start:])
    # Truncate the bf16 region from 16 to 8 bytes, shift everything after.
    new_body = bytearray()
    new_body += body[:lo]
    new_body += bf16_bytes
    new_body += body[hi:]
    # Update header offsets: bf16 hi shrinks by 8; everything after shifts by -8.
    delta = -8
    info["dtype"] = "BF16"
    info["data_offsets"] = [lo, lo + 8]
    for k, v in header.items():
        if k == "bf16" or k == "__metadata__":
            continue
        klo, khi = v["data_offsets"]
        if klo >= hi:
            v["data_offsets"] = [klo + delta, khi + delta]
    new_hdr = json.dumps(header, separators=(",", ":"))
    pad = (-(8 + len(new_hdr.encode("utf-8")))) % 8
    new_hdr += " " * pad
    out = struct.pack("<Q", len(new_hdr.encode("utf-8")))
    out += new_hdr.encode("utf-8")
    out += bytes(new_body)
    path.write_bytes(out)


def manifest_check(out_dir: Path) -> None:
    """Re-open every file we just wrote with the safetensors lib to
    confirm it is well-formed by the reference impl."""
    for f in sorted(out_dir.glob("*.safetensors")):
        with safe_open(str(f), framework="numpy") as s:
            keys = list(s.keys())
            md = s.metadata() or {}
            print(f"{f.name:30s}  tensors={len(keys):2d}  meta_keys={len(md):2d}  size={f.stat().st_size:6d}")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path(__file__).parent / "vectors")
    gen(out)
    manifest_check(out)
