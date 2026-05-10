// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
//
// Profile per-level breakdown of one V-cycle apply on Mire 5k.
// Identifies which phase (smooth / spmv / restrict / prolong /
// bottom) dominates the per-V-cycle wall time. 5 s wall cap.

#include "curvenet/cut_mesh.h"
#include "curvenet/cut_mesh_laplacian.h"
#include "curvenet/halfedge_builder.h"
#include "curvenet/hierarchical_sparsify.h"
#include "curvenet/sparse_linalg.h"
#include "curvenet/vec3.h"
#include "mire_body_data.h"

#include <cmath>
#include <cstdio>
#include <vector>

using curvenet::Vec3;
namespace cm   = curvenet::cut_mesh;
namespace cml  = curvenet::cut_mesh_laplacian;
namespace hscn = curvenet::hsc;
namespace sp   = curvenet::sparse;

int main() {
    std::vector<Vec3> positions;
    positions.reserve(mire_body::n_verts);
    for (int i = 0; i < mire_body::n_verts; ++i) {
        positions.push_back({
            static_cast<double>(mire_body::positions[i * 3 + 0]),
            static_cast<double>(mire_body::positions[i * 3 + 1]),
            static_cast<double>(mire_body::positions[i * 3 + 2]),
        });
    }
    std::vector<int> tris(mire_body::tris,
                            mire_body::tris + mire_body::n_tris * 3);
    const std::size_t nv = positions.size();

    const curvenet::HalfedgeMesh hm =
        curvenet::halfedge_builder::from_triangles(nv, tris);
    cm::CutMesh c;
    c.base = hm;
    c.vertex_kind.assign(nv, cm::CutVertexKind::mesh_vertex_kind());
    c.segment_of_halfedge.assign(hm.he_count(), -1);
    const double mol_delta = cml::default_mollify_delta(positions, tris);
    const sp::SparseMatrixCSR A = cml::assemble_vt_lh_v_csr_robust(
                                     c, positions, mol_delta);
    const hscn::Graph g = hscn::csr_to_graph(A);
    const hscn::Hierarchy h = hscn::build_hierarchy(g);

    std::vector<double> b(nv);
    for (std::size_t i = 0; i < nv; ++i) {
        b[i] = std::sin(0.001 * static_cast<double>(i + 1));
    }
    b = sp::spmv(A, b);

    hscn::VCycleProfile prof;
    const auto x = hscn::v_cycle_apply(h, b, 0, 1, 1, &prof);
    (void)x;

    std::printf("HSC V-cycle profile @ 5k (%zu levels)\n", h.graphs.size());
    std::printf("%-5s %-9s %-7s %-7s %-7s %-7s\n",
                  "lvl", "n", "smooth", "spmv", "restr", "prolong");
    double tot_smooth = 0, tot_spmv = 0, tot_restr = 0, tot_prol = 0, tot_bot = 0, tot_alloc = 0;
    for (std::size_t i = 0; i < prof.smooth_us.size(); ++i) {
        std::printf("%-5zu %-9zu %6.1f  %6.1f  %6.1f  %6.1f\n",
                      i, h.graphs[i].num_verts,
                      prof.smooth_us[i], prof.spmv_us[i],
                      prof.restrict_us[i], prof.prolong_us[i]);
        tot_smooth += prof.smooth_us[i];
        tot_spmv   += prof.spmv_us[i];
        tot_restr  += prof.restrict_us[i];
        tot_prol   += prof.prolong_us[i];
        tot_alloc  += prof.alloc_us[i];
    }
    for (double v : prof.bottom_us) tot_bot += v;
    std::printf("\ntotals (us): smooth=%.1f spmv=%.1f restrict=%.1f prolong=%.1f bottom=%.1f alloc=%.1f\n",
                  tot_smooth, tot_spmv, tot_restr, tot_prol, tot_bot, tot_alloc);
    std::printf("V-cycle total (sum of phases): %.1f us\n",
                  tot_smooth + tot_spmv + tot_restr + tot_prol + tot_bot + tot_alloc);
    return 0;
}
