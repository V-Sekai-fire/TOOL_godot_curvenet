// Real-input validator for Curvenet.SlangCodegen.CheckAscii. Drives the
// slangc-emitted CPU kernel with a deterministic mixed-ASCII / non-ASCII
// byte stream and verifies the output flags array against an independent
// host implementation. Bit-exact equality on every flag slot.
//
// The kernel is the Slang port of cuJSON's `checkAscii` (parse_standard_json.cu)
// rephrased to use per-thread output slots instead of `__shared__ + atomicOr`,
// because slangc -target cpp doesn't support InterlockedOr. The "is the
// whole buffer ASCII?" question reduces to `OR over flags[]` at the host
// side — the kernel itself remains embarrassingly parallel.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "check_ascii_emit.cpp"

static constexpr double kBudgetSeconds = 5.0;

int main() {
    const auto t0 = std::chrono::steady_clock::now();

    // Deterministic input: 16384 uint32 words = 65,536 bytes.
    // Lay down a mix of ASCII regions (bytes 0..127) and non-ASCII
    // regions (top bit set) at known offsets so we can independently
    // assert per-thread output.
    constexpr uint32_t N_WORDS = 16384u;
    constexpr uint32_t WORDS_PER_THREAD = 16u;
    constexpr uint32_t N_THREADS = (N_WORDS + WORDS_PER_THREAD - 1u) / WORDS_PER_THREAD;

    std::vector<uint32_t> data(N_WORDS);
    for (uint32_t i = 0; i < N_WORDS; ++i) {
        // Default: pure-ASCII pattern (each byte < 0x80).
        // Bytes are 'a' + (offset mod 26) interleaved.
        const uint32_t base = i * 4u;
        data[i] =
              (uint32_t)('a' + ((base + 0u) % 26u))        << 0
            | (uint32_t)('a' + ((base + 1u) % 26u))        << 8
            | (uint32_t)('a' + ((base + 2u) % 26u))        << 16
            | (uint32_t)('a' + ((base + 3u) % 26u))        << 24;
    }
    // Seed a few non-ASCII words at predictable positions.
    const std::vector<uint32_t> bad_positions = {0u, 17u, 257u, 4095u, 12345u, N_WORDS - 1u};
    for (uint32_t p : bad_positions) {
        data[p] = 0xE3818182u;  // ≥ one byte ≥ 0x80
    }

    std::vector<uint32_t> flags(N_THREADS, 0u);

    // Host reference: for each thread, OR the words [start, end), then
    // emit 1 iff any byte in the OR has its top bit set.
    std::vector<uint32_t> flags_ref(N_THREADS, 0u);
    for (uint32_t t = 0; t < N_THREADS; ++t) {
        const uint32_t start = t * WORDS_PER_THREAD;
        const uint32_t end   = std::min(start + WORDS_PER_THREAD, N_WORDS);
        uint32_t acc = 0u;
        for (uint32_t j = start; j < end; ++j) acc |= data[j];
        flags_ref[t] = (acc & 0x80808080u) != 0u ? 1u : 0u;
    }

    CheckAsciiParams_0 params{N_WORDS, WORDS_PER_THREAD};
    GlobalParams_0 gp{};
    gp.params_0 = &params;
    gp.data_0.data  = data.data();   gp.data_0.count  = N_WORDS;
    gp.flags_0.data = flags.data();  gp.flags_0.count = N_THREADS;

    // [numthreads(256, 1, 1)] — dispatch ceil(N_THREADS / 256) groups so
    // every output slot the host writes is touched by the kernel and the
    // tail-group bounds check fires.
    const uint32_t groups = (N_THREADS + 255u) / 256u;
    ComputeVaryingInput vi{};
    vi.startGroupID = uint3(0, 0, 0);
    vi.endGroupID   = uint3(groups, 1, 1);
    main_0(&vi, nullptr, &gp);

    int fails = 0;
    for (uint32_t t = 0; t < N_THREADS; ++t) {
        if (flags[t] != flags_ref[t]) {
            if (fails < 5) {
                std::fprintf(stderr,
                    "check_ascii mismatch at thread t=%u: got %u, expected %u\n",
                    t, flags[t], flags_ref[t]);
            }
            ++fails;
        }
    }

    // The final "any non-ASCII?" predicate the cuJSON kernel exposes:
    // OR-reduction over the flag array. Sanity-check that it agrees with
    // the placement of `bad_positions` (which should make several threads
    // light up).
    uint32_t any_set = 0u;
    for (uint32_t v : flags) any_set |= v;
    uint32_t any_set_ref = 0u;
    for (uint32_t v : flags_ref) any_set_ref |= v;

    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(t1 - t0).count();
    if (elapsed > kBudgetSeconds) {
        std::fprintf(stderr,
            "check_ascii: TIMEOUT — %.3fs > budget %.1fs\n",
            elapsed, kBudgetSeconds);
        return 2;
    }
    if (fails == 0 && any_set == 1u && any_set_ref == 1u) {
        std::printf(
            "check_ascii: %u/%u flags OK (n_words=%u, words_per_thread=%u, %u groups, any_set=%u, %.1fms)\n",
            N_THREADS, N_THREADS, N_WORDS, WORDS_PER_THREAD, groups, any_set, elapsed * 1000.0);
        return 0;
    }
    std::fprintf(stderr,
        "check_ascii: %d slot mismatches, any_set=%u (expected 1), ref=%u\n",
        fails, any_set, any_set_ref);
    return 1;
}
