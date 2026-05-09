# Character mesh assets

These are externally-sourced 3D character meshes used to validate the
deformer on real (non-synthetic) topology. They are gitignored — only
this README is tracked.

## Mire (Quest LOD)

[V-Sekai-fire/mesh-mille-mire-feuille](https://github.com/V-Sekai-fire/mesh-mille-mire-feuille)
hosts both the desktop and Quest LOD blend files. We use the Quest
version: 5485 vertices, 9956 triangles, humanoid in T-pose at the
world origin.

To re-fetch:

```sh
mkdir -p demo/character
curl -fsSL -o demo/character/MireQuest.blend \
  https://raw.githubusercontent.com/V-Sekai-fire/mesh-mille-mire-feuille/main/MireQuest.blend
```

To re-bake the vertex data into `tests/mire_body_data.h` (run once,
result is committed):

1. Open Blender with the BlenderMCP addon connected to Claude Code.
2. Run the conversion script via `mcp__blender__execute_blender_code`
   (see the commit that landed `tests/mire_body_data.h` for the exact
   script).

The `.glb` form is also produced by the same Blender step; it's used
by the demo scene when we have one — the C++ bench reads the header
directly and doesn't need the GLB.
