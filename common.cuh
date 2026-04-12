#pragma once
// =============================================================================
//  common.cuh — Shared macros, constants, and device helpers
//  Used by every translation unit in the GA+STOMP project
// =============================================================================

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// ── CUDA error-checking macro ─────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _err = (call);                                              \
        if (_err != cudaSuccess) {                                              \
            fprintf(stderr, "[CUDA ERROR] %s:%d  →  %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(_err));              \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ── Kernel launch dimensions ──────────────────────────────────────────────────
#define BLOCK_SIZE       256      // threads per block for most kernels
#define STATS_BLOCK      256      // threads per block for stats kernel
#define QT_BLOCK         256      // threads per block for QT-init kernel

// ── Numerical constants ───────────────────────────────────────────────────────
#define EPS              1e-8f    // guard against division by zero
#define INF_F            3.4028235e+38f   // float max (initial MP value)

// ── Atomic min for float ──────────────────────────────────────────────────────
// CUDA has no native atomicMin(float*,float); we reinterpret via int CAS.
// Correct for all non-negative floats (distances are always ≥ 0).
__device__ __forceinline__
void atomicMinFloat(float* __restrict__ addr, float val)
{
    int* addr_as_int = reinterpret_cast<int*>(addr);
    int  old_bits    = *addr_as_int;
    int  val_bits    = __float_as_int(val);
    while (val < __int_as_float(old_bits)) {
        int assumed = old_bits;
        old_bits = atomicCAS(addr_as_int, assumed, val_bits);
        if (old_bits == assumed) break;   // CAS succeeded
    }
}

// ── Warp-level min reduction via shuffle ──────────────────────────────────────
// Reduces 32 lanes to lane-0 holding the minimum — no shared memory needed.
__device__ __forceinline__
float warpReduceMin(float val)
{
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xFFFFFFFF, val, offset);
        val = fminf(val, other);
    }
    return val;
}

// ── Host-side Gaussian random (Box-Muller) ────────────────────────────────────
inline float gaussianRand()
{
    float u1 = (float)(rand() + 1) / ((float)RAND_MAX + 2.0f);
    float u2 = (float)rand()        / ((float)RAND_MAX + 1.0f);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

// ── Clamp helper ──────────────────────────────────────────────────────────────
template<typename T>
__host__ __device__ __forceinline__
T clampVal(T x, T lo, T hi) { return (x < lo) ? lo : (x > hi) ? hi : x; }
