#!/usr/bin/env python
"""Slang build glue for SCons.

Adapted from https://github.com/DevPrice/godot-slang/blob/main/build_slang.py
(MIT, K. S. Ernest "iFire" Lee re-spin). Differences from upstream:

- Slang lives at `thirdparty/slang/` (this project's vendor path), not
  `slang/` (godot-slang's submodule path).
- The vendored copy has its `.git` stripped, so version detection
  falls back to "" — the unversioned slang-compiler library name is used.
- The preset chain is the same: configure with `--preset default`,
  build with `--preset releaseWithDebugInfo`.

The function `slang(env, output_dir)` produces a SCons Command that:

1. Runs `cmake --preset default` against `thirdparty/slang/` (one-shot
   if `CMakeCache.txt` is missing).
2. Runs `cmake --build --preset releaseWithDebugInfo`.
3. Installs the resulting `libslang-compiler.{so,dylib,dll}` into
   `output_dir`.

Returns the install command so the caller (`SConstruct`) can `Depends`
the GDExtension link step on it.
"""

import os
import subprocess


SLANG_DIR = "thirdparty/slang"


def slang(env, output_dir, build_preset="default", build_type="releaseWithDebugInfo"):
    def get_slang_version():
        # The vendored Slang has no .git; fall back to unversioned naming.
        # If you switch to a submodule layout, restore the upstream git
        # describe call here.
        return ""

    def build_slang(env, target, source):
        slang_dir = SLANG_DIR
        build_dir = os.path.join(slang_dir, "build")
        cmake_env = os.environ.copy()

        configure_file = os.path.join(build_dir, "CMakeCache.txt")
        if not os.path.exists(configure_file):
            os.makedirs(build_dir, exist_ok=True)
            configure_cmd = [
                "cmake",
                "--preset", build_preset,
                "-DSLANG_ENABLE_GFX=FALSE",
                "-DSLANG_ENABLE_SLANGI=FALSE",
                "-DSLANG_ENABLE_SLANGRT=FALSE",
                "-DSLANG_ENABLE_TESTS=FALSE",
                "-DSLANG_ENABLE_EXAMPLES=FALSE",
                "-DSLANG_LIB_TYPE=SHARED",
            ]
            if env["platform"] == "macos":
                if env["arch"] == "universal":
                    configure_cmd.append("-DCMAKE_OSX_ARCHITECTURES=arm64;x86_64")
                else:
                    configure_cmd.append(f"-DCMAKE_OSX_ARCHITECTURES={env['arch']}")

            result = subprocess.run(configure_cmd, cwd=slang_dir, env=cmake_env)
            if result.returncode != 0:
                print("Error: Slang CMake configuration failed!")
                return result.returncode

        build_cmd = [
            "cmake",
            "--build",
            "--preset", build_type,
        ]
        result = subprocess.run(build_cmd, cwd=slang_dir, env=cmake_env)
        if result.returncode != 0:
            print("Error: Slang CMake build failed!")
            return result.returncode
        return None

    # Source dependency list — files SCons should hash to know whether
    # to invoke the build action.
    slang_sources = [
        os.path.join(SLANG_DIR, "slang-tag-version.h.in"),
        os.path.join(SLANG_DIR, "CMakeLists.txt"),
    ]
    slang_sources += env.Glob(os.path.join(SLANG_DIR, "*", "CMakeLists.txt"))
    slang_sources += env.Glob(os.path.join(SLANG_DIR, "include", "*.h"))
    slang_sources += env.Glob(os.path.join(SLANG_DIR, "source", "slang", "*.cpp"))
    slang_sources += env.Glob(os.path.join(SLANG_DIR, "source", "slang", "*.h"))

    slang_lib_dir = "bin" if env["platform"] == "windows" else "lib"
    slang_lib_path = os.path.join(SLANG_DIR, "build", "RelWithDebInfo", slang_lib_dir)

    if env["platform"] == "linux":
        slang_lib_file = f"{env.subst('$SHLIBPREFIX')}slang-compiler{env['SHLIBSUFFIX']}{get_slang_version()}"
    elif env["platform"] == "macos":
        slang_lib_file = f"{env.subst('$SHLIBPREFIX')}slang-compiler{get_slang_version()}{env['SHLIBSUFFIX']}"
    else:
        slang_lib_file = f"{env.subst('$SHLIBPREFIX')}slang-compiler{env['SHLIBSUFFIX']}"

    slang_lib_output = os.path.join(slang_lib_path, slang_lib_file)
    slang_outputs = [env.File(slang_lib_output)]

    if env["platform"] == "windows":
        slang_outputs += [env.File(os.path.join(slang_lib_path, "slang-compiler.lib"))]

    slang_build = env.Command(
        slang_outputs, slang_sources, env.Action(build_slang, "Building Slang...")
    )

    if env["platform"] in ["linux", "macos"]:
        unversioned = os.path.join(
            slang_lib_path,
            f"{env.subst('$SHLIBPREFIX')}slang-compiler{env['SHLIBSUFFIX']}",
        )

        def create_symlink(target, source, env):
            if not os.path.exists(unversioned):
                os.symlink(slang_lib_file, unversioned)

        symlink_action = env.Command(
            unversioned, slang_lib_output,
            env.Action(create_symlink, "Creating slang symlink..."),
        )
        env.Depends(symlink_action, slang_build)
        env.AlwaysBuild(symlink_action)

    slang_install_command = env.InstallAs(
        os.path.join(output_dir, slang_lib_file), slang_lib_output
    )
    env.Depends(slang_install_command, slang_build)

    return slang_install_command
