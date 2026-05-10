import Lake
open Lake DSL System

package Curvenet where

require LeanSlang from git
  "https://github.com/V-Sekai-fire/lean-slang.git" @ "v0.0.5"

@[default_target] lean_lib Curvenet where
@[default_target] lean_lib LeanGltf where
@[default_target] lean_lib LeanSafetensors where

lean_exe emit_shaders where
  root := `EmitShaders

lean_exe axpy_validate where
  root := `AxpyValidate
  supportInterpreter := true

lean_exe gltf_inspect where
  root := `GltfInspect

lean_exe safetensors_roundtrip where
  root := `SafetensorsRoundtrip

lean_exe safetensors_to_gltf where
  root := `SafetensorsToGltf

lean_exe safetensors_zup_to_yup where
  root := `SafetensorsZupToYup

extern_lib axpy_native pkg := do
  let workDir := pkg.dir / "native"
  let oFile  := pkg.buildDir / "native" / "axpy_shim.o"
  let srcFile := workDir / "axpy_shim.cpp"
  let slangIncl := pkg.dir / ".." / "bin" / ".slang" / "include"
  let leanIncl ← getLeanIncludeDir
  let flags := #[
    "-std=c++17", "-O2", "-fPIC", "-Wno-unused-parameter",
    "-I", slangIncl.toString,
    "-I", workDir.toString,
    "-I", leanIncl.toString
  ]
  let srcJob ← inputTextFile srcFile
  let oJob ← buildO oFile srcJob #[] flags "c++"
  let libName := nameToStaticLib "axpy_native"
  buildStaticLib (pkg.staticLibDir / libName) #[oJob]
