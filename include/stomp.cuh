#pragma once
// =============================================================================
//  stomp.cuh -- GPU-STOMP with diagonal parallelisation
//
//  Two modes controlled by STOMPConfig.approximate:
//
//  Full mode (approximate = false, default):
//    All n_diags diagonals are processed in order.
//    Produces the exact Matrix Profile.
//    Complexity: O(n^2)
//
//  Approximate mode (approximate = true):
//    Only floor(approx_frac * n_diags) diagonals are processed, chosen in
//    a deterministic strided pattern that spreads coverage evenly.
//    Produces a valid lower bound: MP[i] <= true_MP[i] everywhere.
//    Complexity: O(approx_frac * n^2)
//    Use for n > 50000 or during early GA generations (fidelity ladder).
//
//  MPI recovery (recover_indices = true):
//    After MP is computed, a second CPU pass identifies which diagonal
//    produced the minimum for each profile position, filling MPI[].
//    Skip this during GA evaluation (only fitness values matter).
//    Always enable for the final best-individual run.
// =============================================================================

#include "common.cuh"

// -----------------------------------------------------------------------------
//  Configuration -- one instance per GA individual
// -----------------------------------------------------------------------------
struct STOMPConfig {
    int   window_size;          // m -- subsequence length
    float ez_factor;            // exclusion_zone = floor(ez_factor * m)
    int   min_motif_count;      // passed to fitness; not used by kernel
    bool  approximate;          // use strided-diagonal approximation
    float approx_frac;          // fraction of diagonals to process [0.1, 1.0]
    bool  recover_indices;      // fill MPI[] after MP is computed

    STOMPConfig()
        : window_size(50), ez_factor(0.25f), min_motif_count(5),
          approximate(false), approx_frac(1.0f), recover_indices(false) {}
};

// -----------------------------------------------------------------------------
//  Result -- returned to host
// -----------------------------------------------------------------------------
struct STOMPResult {
    float* mp;              // [profile_len] Matrix Profile distances (host)
    int*   mpi;             // [profile_len] Matrix Profile indices   (host)
    int    profile_len;     // L = n - m + 1
    float  kernel_ms;       // GPU kernel wall time (ms)
    float  precompute_ms;   // stats + QT-init kernel time (ms)
    float  total_ms;        // upload + compute + download (ms)
};

// -----------------------------------------------------------------------------
//  Main diagonal STOMP kernel
// -----------------------------------------------------------------------------
__global__ void stomp_diagonal_kernel(
    const float* __restrict__ T,
    const float* __restrict__ means,
    const float* __restrict__ stds,
    const float* __restrict__ QT_init,
    float*       MP,
    int*         MPI,
    int n, int m, int ez
);

// Approximate variant: one CUDA thread per strided diagonal
__global__ void stomp_approx_kernel(
    const float* __restrict__ T,
    const float* __restrict__ means,
    const float* __restrict__ stds,
    const float* __restrict__ QT_init,
    float*       MP,
    int*         MPI,
    int n, int m, int ez,
    int stride      // process diagonals ez, ez+stride, ez+2*stride, ...
);

// -----------------------------------------------------------------------------
//  Host launcher
// -----------------------------------------------------------------------------
STOMPResult run_stomp(
    const float*       h_T,
    int                n,
    const STOMPConfig& cfg
);
