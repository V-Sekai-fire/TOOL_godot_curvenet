// Hand-written minimal LEMON config for cage_deformer / curvenet.
// Only the matching code path is exercised — no LP/MIP solvers vendored.
#ifndef LEMON_CONFIG_H
#define LEMON_CONFIG_H

#define LEMON_VERSION "1.3.1-vendored"

#define LEMON_HAVE_LONG_LONG 1
#define LEMON_CXX11 1

#if defined(_WIN32) || defined(_WIN64)
#define LEMON_WIN32 1
#define LEMON_USE_WIN32_THREADS 1
#else
#define LEMON_USE_PTHREAD 1
#endif

#endif
