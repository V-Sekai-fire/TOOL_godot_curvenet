import Lake
open Lake DSL System

package Curvenet where

require LeanSlang from git
  "https://github.com/V-Sekai-fire/lean-slang.git" @ "v0.0.5"

@[default_target] lean_lib Curvenet where
  precompileModules := true
@[default_target] lean_lib LeanGltf where

lean_exe emit_shaders where
  root := `EmitShaders

lean_exe axpy_validate where
  root := `AxpyValidate
  supportInterpreter := true

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
