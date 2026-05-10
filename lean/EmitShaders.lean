import LeanSlang
import Curvenet.SlangCodegen

/-!
# `emit_shaders` — Lake exe dumping every Slang shader to disk

For each `Curvenet.SlangCodegen.*` kernel, writes the emitted Slang
source to `<outDir>/<name>.slang`. Used as input to slangc for
syntactic / semantic validation beyond the `native_decide` text
fixtures.

Usage:

    lake exe emit_shaders /path/to/output/dir
-/

open Curvenet.SlangCodegen
open LeanSlang

private def kernels : List (String × SlangShaderModule) :=
  [ ("direct_delta_mush", DirectDeltaMush.shader)
  , ("axpy",              Axpy.shader)
  , ("axpy_multi",        AxpyMulti.shader)
  , ("saxpby",            Saxpby.shader)
  , ("saxpby_multi",      SaxpbyMulti.shader)
  , ("jacobi",            Jacobi.shader)
  , ("jacobi_multi",      JacobiMulti.shader)
  , ("spmv",              Spmv.shader)
  , ("spmv_multi",        SpmvMulti.shader)
  , ("dot_reduce",        DotReduce.shader)
  , ("dot_reduce_multi",  DotReduceMulti.shader)
  , ("sgs_color",         SgsColor.shader)
  , ("polygon_laplacian", PolygonLaplacian.shader)
  , ("robust_laplacian",  RobustLaplacian.shader)
  , ("scaled_frames",     ScaledFrames.shader)
  , ("segment_gradient",  SegmentGradient.shader)
  , ("intersection_frames", IntersectionFrames.shader)
  , ("curve_interp",      CurveInterp.shader)
  , ("vec3",              Vec3.shader)
  ]

def main (args : List String) : IO UInt32 := do
  let outDir := args.headD "."
  IO.FS.createDirAll outDir
  for (name, m) in kernels do
    let path := outDir ++ "/" ++ name ++ ".slang"
    IO.FS.writeFile path (LeanSlang.emit m ++ "\n")
    IO.println s!"wrote {path}"
  return 0
