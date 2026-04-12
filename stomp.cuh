#pragma once
// =============================================================================
//  stomp.cuh — GPU-STOMP with diagonal parallelisation
//
//  Core algorithm reference:
//    Zhu et al. "Matrix Profile II: Exploiting a Novel Algorithm and GPUs
//    to Break the One Hundred Million Barrier for Time Series Motifs and Joins"
//    (ICDM 2016 / Zimmerman GPU extension 2019)
//
//  Parallelisation strategy:
//    ┌────────────────────────────────────────────────────────┐
//    │  Each CUDA thread handles one diagonal d independently │
//    │  No inter-thread dependency → embarrassingly parallel  │
//    │  Threads update MP[i] and MP[j] with atomicMinFloat    │
//    └────────────────────────────────────────────────────────┘
//
//  Memory placement:
//    Global : T, means, stds, QT_init, MP  (large, persistent)
//    Register: qt, mu_i/j, sig_i/j, dist, mp_running  (hot loop vars)
//    Read-only cache (__ldg): T, means, stds  (never written in kernel)
// =============================================================================

#include "common.cuh"

// ── Configuration passed per GA individual ───────────────────────────────────
struct STOMPConfig {
    int   window_size;          // m  — subsequence length
    float ez_factor;            // exclusion_zone = floor(ez_factor * m)
    int   min_motif_count;      // used downstream by fitness; not by kernel
};

// ── Output returned to host after one STOMP run ──────────────────────────────
struct STOMPResult {
    float* mp;    // [profile_len] Matrix Profile distances (host-side)
    int*   mpi;   // [profile_len] Matrix Profile indices  (host-side)
    int    profile_len;         // L = n - m + 1
};

// ── Main diagonal STOMP kernel ────────────────────────────────────────────────
// One thread per diagonal d.
// Inner loop slides QT along the diagonal using the STOMP recurrence:
//   QT(i,j) = QT(i-1,j-1) - T[i-1]*T[j-1] + T[i+m-1]*T[j+m-1]
// Then converts to z-normalised Euclidean distance and atomically updates MP.
__global__ void stomp_diagonal_kernel(
    const float* __restrict__ T,        // [n]
    const float* __restrict__ means,    // [L]
    const float* __restrict__ stds,     // [L]
    const float* __restrict__ QT_init,  // [L - ez]  diagonal seeds
    float*       MP,                    // [L]  matrix profile  (atomic writes)
    int*         MPI,                   // [L]  profile index   (atomic writes)
    int n,
    int m,
    int ez                              // exclusion zone width
);

// ── Host launcher ─────────────────────────────────────────────────────────────
// Uploads T, runs precompute + diagonal kernel, downloads result.
// Caller is responsible for free()-ing result.mp and result.mpi.
STOMPResult run_stomp(
    const float*       h_T,    // [n]  host time series
    int                n,
    const STOMPConfig& cfg
);
