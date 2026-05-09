// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
//
// Phase-0 meshlet decomposition diagnostic. Per the locked-in 70k+
// vertex target, the deformer will need to do constant work per
// meshlet rather than scaling with total mesh size. Step 1 is
// confirming that meshoptimizer's `meshopt_buildMeshlets` produces
// reasonable clusters on our actual mesh.
//
// Reports:
//   * meshlet count, vertex/triangle counts per meshlet
//   * shared-vertex (boundary) count: verts that appear in >1 meshlet
//   * histogram of meshlet sizes
//
// Run via `make -C tests diag_meshlet`. No solver work yet — this
// just validates the partition.

#include "mire_body_data.h"

#include "meshoptimizer.h"

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <unordered_set>
#include <vector>

namespace {

struct MeshletReport {
    std::size_t n_meshlets;
    std::size_t boundary_verts;
    std::size_t min_v, max_v;
    std::size_t min_t, max_t;
    double mean_v, mean_t;
};

MeshletReport run(std::size_t max_verts, std::size_t max_tris,
                    const std::vector<unsigned int> &indices,
                    const std::vector<float> &positions,
                    std::size_t n_verts) {
    const std::size_t n_indices = indices.size();
    const std::size_t bound = meshopt_buildMeshletsBound(n_indices, max_verts, max_tris);

    std::vector<meshopt_Meshlet> meshlets(bound);
    std::vector<unsigned int>    meshlet_vertices (bound * max_verts);
    std::vector<unsigned char>   meshlet_triangles(bound * max_tris * 3);

    const float cone_weight = 0.0f; // we don't care about render cone-culling
    const std::size_t k = meshopt_buildMeshlets(
        meshlets.data(),
        meshlet_vertices.data(),
        meshlet_triangles.data(),
        indices.data(), n_indices,
        positions.data(), n_verts,
        sizeof(float) * 3,
        max_verts, max_tris,
        cone_weight);

    meshlets.resize(k);

    MeshletReport r{};
    r.n_meshlets = k;
    r.min_v = SIZE_MAX; r.max_v = 0;
    r.min_t = SIZE_MAX; r.max_t = 0;
    std::size_t total_v = 0, total_t = 0;
    std::vector<std::uint8_t> seen_count(n_verts, 0);

    for (const auto &ml : meshlets) {
        const std::size_t mv = ml.vertex_count;
        const std::size_t mt = ml.triangle_count;
        if (mv < r.min_v) r.min_v = mv;
        if (mv > r.max_v) r.max_v = mv;
        if (mt < r.min_t) r.min_t = mt;
        if (mt > r.max_t) r.max_t = mt;
        total_v += mv;
        total_t += mt;
        for (std::size_t j = 0; j < mv; ++j) {
            const unsigned int v = meshlet_vertices[ml.vertex_offset + j];
            if (seen_count[v] < 255) ++seen_count[v];
        }
    }
    r.mean_v = static_cast<double>(total_v) / k;
    r.mean_t = static_cast<double>(total_t) / k;
    r.boundary_verts = 0;
    for (std::uint8_t s : seen_count) if (s > 1) ++r.boundary_verts;
    return r;
}

void print_report(const char *label, const MeshletReport &r, std::size_t n_verts) {
    std::printf("  %s: %zu meshlets\n", label, r.n_meshlets);
    std::printf("    verts/meshlet:  min=%zu  max=%zu  mean=%.1f\n",
                  r.min_v, r.max_v, r.mean_v);
    std::printf("    tris/meshlet:   min=%zu  max=%zu  mean=%.1f\n",
                  r.min_t, r.max_t, r.mean_t);
    std::printf("    shared (boundary) verts: %zu / %zu  (%.1f%%)\n",
                  r.boundary_verts, n_verts,
                  100.0 * r.boundary_verts / n_verts);
}

} // namespace

int main(int argc, char **argv) {
    (void)argc; (void)argv;

    const std::size_t n_verts = mire_body::n_verts;
    const std::size_t n_tris  = mire_body::n_tris;
    std::printf("Mire body: %zu verts, %zu tris\n\n", n_verts, n_tris);

    // meshoptimizer takes flat float positions and uint indices.
    std::vector<float> positions(mire_body::positions, mire_body::positions + n_verts * 3);
    std::vector<unsigned int> indices(n_tris * 3);
    for (std::size_t i = 0; i < n_tris * 3; ++i) {
        indices[i] = static_cast<unsigned int>(mire_body::tris[i]);
    }

    // Sweep a few meshlet size caps to see what the partition looks like.
    print_report("max_verts=64,  max_tris=124",
                  run(64, 124, indices, positions, n_verts), n_verts);
    print_report("max_verts=128, max_tris=124",
                  run(128, 124, indices, positions, n_verts), n_verts);
    print_report("max_verts=256, max_tris=512",
                  run(256, 512, indices, positions, n_verts), n_verts);

    // Extrapolation: at 70k verts, the same per-meshlet size cap yields
    // roughly n_verts / max_v meshlets, with the same per-meshlet CG cost.
    std::printf("\nExtrapolation to 70k verts (assuming the same vert/meshlet ratio):\n");
    const MeshletReport r256 = run(256, 512, indices, positions, n_verts);
    if (r256.mean_v > 0.0) {
        const double k_at_70k = 70000.0 / r256.mean_v;
        std::printf("  ~%.0f meshlets, ~%.1f verts each\n", k_at_70k, r256.mean_v);
    }
    return 0;
}
