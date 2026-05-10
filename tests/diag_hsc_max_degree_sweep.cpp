// Sweep max_degree to find sweet spot of (iters, V-cycle cost).
#include "curvenet/cut_mesh.h"
#include "curvenet/cut_mesh_laplacian.h"
#include "curvenet/halfedge_builder.h"
#include "curvenet/hierarchical_sparsify.h"
#include "curvenet/sparse_linalg.h"
#include "curvenet/vec3.h"
#include "mire_body_data.h"
#include <chrono>
#include <cmath>
#include <cstdio>

using curvenet::Vec3;
namespace cm   = curvenet::cut_mesh;
namespace cml  = curvenet::cut_mesh_laplacian;
namespace hscn = curvenet::hsc;
namespace sp   = curvenet::sparse;

double now_ms() {
    using namespace std::chrono;
    return duration_cast<duration<double, std::milli>>(steady_clock::now().time_since_epoch()).count();
}

int main() {
    std::vector<Vec3> positions;
    positions.reserve(mire_body::n_verts);
    for (int i = 0; i < mire_body::n_verts; ++i) {
        positions.push_back({ static_cast<double>(mire_body::positions[i*3+0]),
                                static_cast<double>(mire_body::positions[i*3+1]),
                                static_cast<double>(mire_body::positions[i*3+2]) });
    }
    std::vector<int> tris(mire_body::tris, mire_body::tris + mire_body::n_tris*3);
    const std::size_t nv = positions.size();
    const curvenet::HalfedgeMesh hm = curvenet::halfedge_builder::from_triangles(nv, tris);
    cm::CutMesh c; c.base = hm;
    c.vertex_kind.assign(nv, cm::CutVertexKind::mesh_vertex_kind());
    c.segment_of_halfedge.assign(hm.he_count(), -1);
    const double mol = cml::default_mollify_delta(positions, tris);
    const sp::SparseMatrixCSR A = cml::assemble_vt_lh_v_csr_robust(c, positions, mol);
    const hscn::Graph g = hscn::csr_to_graph(A);
    std::vector<double> y(nv);
    for (std::size_t i = 0; i < nv; ++i) y[i] = std::sin(0.001 * static_cast<double>(i + 1));
    const std::vector<double> b = sp::spmv(A, y);
    const std::vector<double> x0(nv, 0.0);

    std::printf("%-6s %-8s %-7s %-10s %-7s\n", "maxdeg", "build_ms", "iters", "solve_ms", "total");
    for (std::size_t md : { 16, 24, 32, 48, 64, 96, 128 }) {
        const double t0 = now_ms();
        const hscn::Hierarchy h = hscn::build_hierarchy(g, 64, 64, 0.0, md);
        const double t_build = now_ms() - t0;
        std::size_t iters = 0;
        const double t1 = now_ms();
        const auto x = hscn::cg_hsc_with_guess(A, h, b, x0, 1000, 1e-8, &iters);
        (void)x;
        const double t_solve = now_ms() - t1;
        std::printf("%-6zu %-8.1f %-7zu %-10.2f %-7.1f\n",
                      md, t_build, iters, t_solve, t_build + t_solve);
    }
    return 0;
}
