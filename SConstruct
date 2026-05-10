#!/usr/bin/env python
import os

from build_slang import slang


def normalize_path(val, env):
    return val if os.path.isabs(val) else os.path.join(env.Dir("#").abspath, val)


def validate_parent_dir(key, val, env):
    if not os.path.isdir(normalize_path(os.path.dirname(val), env)):
        raise UserError("'%s' is not a directory: %s" % (key, os.path.dirname(val)))


libname = "tris_to_quads"
projectdir = "demo"

localEnv = Environment(tools=["default"], PLATFORM="")

customs = ["custom.py"]
customs = [os.path.abspath(path) for path in customs]

opts = Variables(customs, ARGUMENTS)
opts.Add(
    BoolVariable(
        key="compiledb",
        help="Generate compilation DB (`compile_commands.json`) for external tools",
        default=localEnv.get("compiledb", False),
    )
)
opts.Add(
    PathVariable(
        key="compiledb_file",
        help="Path to a custom `compile_commands.json` file",
        default=localEnv.get("compiledb_file", "compile_commands.json"),
        validator=validate_parent_dir,
    )
)
opts.Add(
    BoolVariable(
        key="build_slang",
        help="Build the vendored Slang shader compiler at thirdparty/slang/ "
             "and link the GDExtension against libslang-compiler. "
             "Adds 5-15 minutes to a cold build; cached after that.",
        default=False,
    )
)
opts.Update(localEnv)

Help(opts.GenerateHelpText(localEnv))

env = localEnv.Clone()
env["compiledb"] = False

env.Tool("compilation_db")
compilation_db = env.CompilationDatabase(
    normalize_path(localEnv["compiledb_file"], localEnv)
)
env.Alias("compiledb", compilation_db)

env = SConscript("thirdparty/godot-cpp/SConstruct", {"env": env, "customs": customs})

env.Append(CPPPATH=["src/"])
# Curvenet runtime is header-only under src/curvenet/. Top-level src/*.cpp
# is the GDExtension binding layer (register_types, node bindings,
# editor plugins).
sources = Glob("src/*.cpp")

# Template builds (template_debug, template_release) drop editor sources
# entirely. Defining TOOLS_ENABLED for the editor target lets shared
# files (`register_types.cpp`, `vertex_handles_3d.cpp`) include editor
# headers + register editor-tier classes when, and only when, the
# binary will run in an editor process.
is_editor = env["target"] == "editor"
if is_editor:
    env.Append(CPPDEFINES=["TOOLS_ENABLED"])
else:
    sources = [
        s for s in sources
        if "editor_plugin" not in str(s) and "gizmo_plugin" not in str(s)
    ]

if env["target"] in ["editor", "template_debug"]:
    try:
        doc_data = env.GodotCPPDocData(
            "src/gen/doc_data.gen.cpp", source=Glob("doc_classes/*.xml")
        )
        sources.append(doc_data)
    except AttributeError:
        print("Not including class reference as we're targeting a pre-4.3 baseline.")

file = "{}{}{}".format(libname, env["suffix"], env["SHLIBSUFFIX"])
filepath = ""

if env["platform"] == "macos" or env["platform"] == "ios":
    file = "{}.{}.{}".format(libname, env["platform"], env["target"])
    # macOS framework bundles require the directory name to match the inner
    # binary name (minus `.framework`) so dlopen can resolve the bundle.
    # SCons adds the `lib` SHLIBPREFIX to the binary, so we mirror that here.
    filepath = "lib{}.framework/".format(file)

libraryfile = "bin/{}/{}{}".format(env["platform"], filepath, file)
library = env.SharedLibrary(
    libraryfile,
    source=sources,
)

copy = env.InstallAs(
    "{}/bin/{}/{}lib{}".format(projectdir, env["platform"], filepath, file), library
)

# Slang integration. Off by default; opt in with `scons build_slang=true` once
# the deformer's runtime DDM matvec is wired to dispatch a Slang-compiled
# compute shader instead of the CPU lbs_matvec path. The pattern follows
# DevPrice/godot-slang: drop libslang-compiler.{so,dylib,dll} into
# `<projectdir>/bin/<platform>/` next to the GDExtension binary.
slang_install = None
if localEnv.get("build_slang", False):
    slang_output_dir = "{}/bin/{}/".format(projectdir, env["platform"])
    slang_install = slang(env, slang_output_dir)
    env.Depends(library, slang_install)

default_args = [library, copy]
if slang_install is not None:
    default_args.append(slang_install)
if localEnv.get("compiledb", False):
    default_args += [compilation_db]
Default(*default_args)
