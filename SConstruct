#!/usr/bin/env python
import os


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

env.Append(CPPPATH=["src/", "thirdparty/lemon/"])
# godot-cpp builds with -fno-exceptions; LEMON's alteration_notifier and edge_set
# use try/throw for listener-rollback on partial failure. Our usage never triggers
# those paths, so we patch the headers (see thirdparty/lemon/patches/no_exceptions.patch)
# to gate the exception code with #ifndef LEMON_NO_EXCEPTIONS, then define it here.
env.Append(CPPDEFINES=["LEMON_NO_EXCEPTIONS"])
# Curvenet math + tris_to_quads converter live under src/curvenet/.
# Top-level src/*.cpp is the GDExtension binding layer (register_types, node bindings).
sources = Glob("src/*.cpp") + Glob("src/curvenet/*.cpp")

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

default_args = [library, copy]
if localEnv.get("compiledb", False):
    default_args += [compilation_db]
Default(*default_args)
