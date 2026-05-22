#pragma once
// =============================================================================
//  utils.cuh -- Subsequence statistics and QT initialisation
//
//  Provides two GPU-accelerated pre-computation passes that STOMP needs
//  before its main diagonal kernel can execute:
//
//    1. compute_stats  -- per-subsequence mean and std (O(nm) parallel)
//    2. compute_qt_init -- QT seed for each diagonal   (O(nm) parallel)
//
//  Both are O(nm) in total work, dominated by the O(n^2) STOMP main loop.
// =============================================================================

#include "common.cuh"

// -- Device kernels (declared here, defined in utils.cu) ----------------------

// One thread per subsequence.  Each thread sums its m elements for mean,
// then computes variance via E[X^2] - E[X]^2 in a single pass.
__global__ void compute_stats_kernel(
    const float* __restrict__ T,    // [n]   time series
    float*       means,             // [L]   output: subsequence means
    float*       stds,              // [L]   output: subsequence stds (sigma >= EPS)
    int n,                          // series length
    int m                           // window size
);

// One thread per diagonal d in [ez, L).
// Computes QT_init[d - ez] = Sum_{k=0}^{m-1} T[k] * T[k+d]  (dot product)
// which seeds the sliding QT update in the STOMP diagonal kernel.
__global__ void compute_qt_init_kernel(
    const float* __restrict__ T,    // [n]
    float*       QT_init,           // [L - ez]  output: diagonal seeds
    int n,
    int m,
    int ez                          // exclusion zone width (skip trivial diags)
);

// -- Host launcher (utils.cu) -------------------------------------------------
// Allocates d_means, d_stds, d_QT_init on the device, fills them, and
// returns the device pointers to the caller.
void precompute(
    const float* d_T,           // [n]  device pointer (already uploaded)
    int          n,
    int          m,
    int          ez,
    float**      d_means_out,   // allocated inside, caller must cudaFree
    float**      d_stds_out,
    float**      d_QT_init_out
);
