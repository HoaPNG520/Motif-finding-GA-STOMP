// =============================================================================
//  utils.cu -- Pre-computation kernels for STOMP
//
//  compute_stats_kernel : O(nm) -- one thread per subsequence
//  compute_qt_init_kernel: O(nm) -- one thread per diagonal
//
//  Both are dominated by the O(n^2) STOMP kernel; they add negligible overhead.
//
//  Memory notes:
//    T     -> read via __ldg (read-only data cache, bypasses L1)
//    means/stds -> written once, never re-read in these kernels
//    All accesses are coalesced: thread i reads T[i], T[i+1], ..., T[i+m-1]
//      which are adjacent -- fits in consecutive 128-byte cache lines.
// =============================================================================

#include "utils.cuh"
#include <cstdio>

// -----------------------------------------------------------------------------
//  Kernel 1 -- Per-subsequence mean and standard deviation
// -----------------------------------------------------------------------------
//  Uses single-pass E[X] and E[X^2] to avoid a second global-memory sweep.
//  Numerical note: var = E[X^2] - E[X]^2  is susceptible to catastrophic
//  cancellation for large-mean signals.  For typical z-normalised inputs this
//  is fine; for production code consider Welford's online algorithm.
__global__ void compute_stats_kernel(
    const float* __restrict__ T,
    float*       means,
    float*       stds,
    int n, int m)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int L = n - m + 1;                // profile length
    if (i >= L) return;

    float sum  = 0.0f;
    float sum2 = 0.0f;

    // Inner loop -- reads m consecutive floats starting at T[i]
    // Thread i reads T[i..i+m-1]; thread i+1 reads T[i+1..i+m].
    // The overlap is m-1 elements: full warp fetches cover a width of
    // BLOCK_SIZE + m - 1 unique values, so the effective BW saving via
    // shared memory would be ~mx.  We rely on L1 / read-only cache here.
    for (int k = 0; k < m; k++) {
        float v = __ldg(&T[i + k]);
        sum  += v;
        sum2 += v * v;
    }

    float mean = sum  / (float)m;
    float var  = sum2 / (float)m - mean * mean;
    var = fmaxf(0.0f, var);           // clamp rounding noise to >= 0

    means[i] = mean;
    stds[i]  = sqrtf(var) + EPS;      // + EPS prevents divide-by-zero in kernel
}

// -----------------------------------------------------------------------------
//  Kernel 2 -- QT seed for each diagonal
// -----------------------------------------------------------------------------
//  QT_init[d - ez] = Sum_{k=0}^{m-1} T[k] * T[k + d]
//
//  This is the dot product between the FIRST subsequence (k=0..m-1) and the
//  subsequence starting at d.  STOMP's recurrence then slides it forward:
//    QT(i, i+d) = QT(i-1, i-1+d) - T[i-1]*T[i-1+d] + T[i+m-1]*T[i+d+m-1]
//
//  A faster alternative is MASS (FFT convolution, O(n log n) total) but the
//  O(nm) approach is simpler and dominated by the O(n^2) STOMP step anyway.
__global__ void compute_qt_init_kernel(
    const float* __restrict__ T,
    float*       QT_init,
    int n, int m, int ez)
{
    int d = blockIdx.x * blockDim.x + threadIdx.x + ez;  // diagonal index
    int L = n - m + 1;
    if (d >= L) return;

    float qt = 0.0f;
    for (int k = 0; k < m; k++) {
        qt += __ldg(&T[k]) * __ldg(&T[k + d]);
    }
    QT_init[d - ez] = qt;
}

// -----------------------------------------------------------------------------
//  Host launcher -- allocates device buffers and runs both kernels
// -----------------------------------------------------------------------------
void precompute(
    const float* d_T,
    int n, int m, int ez,
    float** d_means_out,
    float** d_stds_out,
    float** d_QT_init_out)
{
    int L          = n - m + 1;           // profile length
    int n_diags    = L - ez;              // valid diagonals

    // Allocate
    CUDA_CHECK(cudaMalloc(d_means_out,   L       * sizeof(float)));
    CUDA_CHECK(cudaMalloc(d_stds_out,    L       * sizeof(float)));
    CUDA_CHECK(cudaMalloc(d_QT_init_out, n_diags * sizeof(float)));

    // Kernel 1 -- stats
    {
        dim3 block(STATS_BLOCK);
        dim3 grid((L + STATS_BLOCK - 1) / STATS_BLOCK);
        compute_stats_kernel<<<grid, block>>>(
            d_T, *d_means_out, *d_stds_out, n, m);
        CUDA_CHECK(cudaGetLastError());
    }

    // Kernel 2 -- QT seeds
    {
        dim3 block(QT_BLOCK);
        dim3 grid((n_diags + QT_BLOCK - 1) / QT_BLOCK);
        compute_qt_init_kernel<<<grid, block>>>(
            d_T, *d_QT_init_out, n, m, ez);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaDeviceSynchronize());
}
