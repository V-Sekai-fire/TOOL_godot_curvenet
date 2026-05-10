#!/usr/bin/env python3
"""Convert tests/mire_body_70k_data.h into raw float32/int32 binaries
that the `mire_to_gltf` Lake exe consumes.

Usage:

    misc/mire_h_to_bin.py tests/mire_body_70k_data.h /tmp/mire

Writes:

    /tmp/mire/positions.f32 — n_verts * 3 little-endian fp32 values
    /tmp/mire/tris.i32      — n_tris * 3 little-endian uint32 indices

The header is auto-generated from MireQuest.blend; the only real work
here is parsing the C++ array literals out of it.
"""

import os
import re
import struct
import sys
from pathlib import Path


def parse_array(text: str, ident: str) -> list[float]:
    m = re.search(
        r"\binline const \w+ " + re.escape(ident) + r"\[\]\s*=\s*\{(.*?)\};",
        text, flags=re.DOTALL,
    )
    if not m:
        raise SystemExit(f"identifier '{ident}' not found in header")
    body = m.group(1)
    body = re.sub(r"//[^\n]*", "", body)
    tokens = [t for t in re.split(r"[\s,]+", body) if t]
    out = []
    for t in tokens:
        if t.endswith("f"):
            out.append(float(t[:-1]))
        else:
            out.append(float(t))
    return out


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    src = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    text = src.read_text()
    pos = parse_array(text, "positions")
    tris = parse_array(text, "tris")
    n_verts = len(pos) // 3
    n_tris = len(tris) // 3

    # Sanity-check against the n_verts / n_tris constants in the header.
    decl = re.search(r"n_verts\s*=\s*(\d+)", text)
    if decl and int(decl.group(1)) != n_verts:
        print(f"warning: n_verts mismatch ({decl.group(1)} vs {n_verts})",
              file=sys.stderr)
    decl = re.search(r"n_tris\s*=\s*(\d+)", text)
    if decl and int(decl.group(1)) != n_tris:
        print(f"warning: n_tris mismatch ({decl.group(1)} vs {n_tris})",
              file=sys.stderr)

    pos_path = out_dir / "positions.f32"
    tri_path = out_dir / "tris.i32"
    with open(pos_path, "wb") as f:
        f.write(struct.pack(f"<{len(pos)}f", *pos))
    with open(tri_path, "wb") as f:
        f.write(struct.pack(f"<{len(tris)}I", *(int(t) for t in tris)))

    print(f"wrote {pos_path} ({n_verts} verts, {pos_path.stat().st_size} bytes)")
    print(f"wrote {tri_path} ({n_tris} tris,  {tri_path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
