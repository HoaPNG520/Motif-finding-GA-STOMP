// =============================================================================
//  stomp.cu -- GPU-STOMP diagonal kernel + approximate mode + MPI recovery
// =============================================================================

#include "stomp.cuh"
#include "utils.cuh"
#include "timer.cuh"
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <vector>

// -----------------------------------------------------------------------------
//  Full diagonal kernel -- one thread per diagonal
// -----------------------------------------------------------------------------
__global__ void stomp_diagonal_kernel(
    const float* __restrict__ T,
    const float* __restrict__ means,
    const float* __restrict__ stds,
    const float* __restrict__ QT_init,
    float* MP,
    int*   MPI,
    int n, int m, int ez)
{
    int d = blockIdx.x * blockDim.x + threadIdx.x + ez;
    int L = n - m + 1;
    if (d >= L) return;

    float qt      = __ldg(&QT_init[d - ez]);
    int   diag_len = L - d;

    for (int i = 0; i < diag_len; i++) {
        int j = i + d;

        if (i > 0) {
            qt += __ldg(&T[i + m - 1]) * __ldg(&T[j + m - 1])
                - __ldg(&T[i - 1])     * __ldg(&T[j - 1]);
        }

        float mu_i  = __ldg(&means[i]);
        float mu_j  = __ldg(&means[j]);
        float sig_i = __ldg(&stds[i]);
        float sig_j = __ldg(&stds[j]);

        float sig_prod = sig_i * sig_j;
        float dist;
        if (sig_prod < EPS) {
            dist = 0.0f;
        } else {
            float pearson = (qt / (float)m - mu_i * mu_j) / sig_prod;
            pearson = fminf(1.0f, fmaxf(-1.0f, pearson));
            dist = sqrtf(2.0f * (float)m * (1.0f - pearson));
        }

        atomicMinFloat(&MP[i], dist);
        atomicMinFloat(&MP[j], dist);
    }
}

// -----------------------------------------------------------------------------
//  Approximate kernel -- processes every `stride`-th diagonal
//  Used when cfg.approximate == true (large n or early GA generations)
// -----------------------------------------------------------------------------
__global__ void stomp_approx_kernel(
    const float* __restrict__ T,
    const float* __restrict__ means,
    const float* __restrict__ stds,
    const float* __restrict__ QT_init,
    float* MP,
    int*   MPI,
    int n, int m, int ez, int stride)
{
    // Thread t processes diagonal:  ez + t * stride
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int d = ez + t * stride;
    int L = n - m + 1;
    if (d >= L) return;

    // Recompute QT seed for this diagonal from scratch
    // (QT_init only covers every diagonal when stride==1)
    float qt = 0.0f;
    for (int k = 0; k < m; k++)
        qt += __ldg(&T[k]) * __ldg(&T[k + d]);

    int diag_len = L - d;
    for (int i = 0; i < diag_len; i++) {
        int j = i + d;
        if (i > 0) {
            qt += __ldg(&T[i + m - 1]) * __ldg(&T[j + m - 1])
                - __ldg(&T[i - 1])     * __ldg(&T[j - 1]);
        }

        float mu_i  = __ldg(&means[i]);
        float mu_j  = __ldg(&means[j]);
        float sig_i = __ldg(&stds[i]);
        float sig_j = __ldg(&stds[j]);
        float sig_prod = sig_i * sig_j;
        float dist;
        if (sig_prod < EPS) {
            dist = 0.0f;
        } else {
            float pearson = (qt / (float)m - mu_i * mu_j) / sig_prod;
            pearson = fminf(1.0f, fmaxf(-1.0f, pearson));
            dist = sqrtf(2.0f * (float)m * (1.0f - pearson));
        }
        atomicMinFloat(&MP[i], dist);
        atomicMinFloat(&MP[j], dist);
    }
}

// -----------------------------------------------------------------------------
//  MPI recovery -- CPU pass after MP is downloaded
//  For each position i, finds which diagonal produced the minimum MP[i].
// -----------------------------------------------------------------------------
static void recover_mpi_cpu(
    const float* h_T,
    const float* h_mp,
    int*         h_mpi,
    int n, int m, int ez)
{
    int L = n - m + 1;
    for (int i = 0; i < L; i++) h_mpi[i] = -1;

    for (int d = ez; d < L; d++) {
        int diag_len = L - d;
        double qt = 0.0;
        for (int k = 0; k < m; k++) qt += (double)h_T[k] * h_T[k + d];

        for (int i = 0; i < diag_len; i++) {
            int j = i + d;
            if (i > 0) {
                qt += (double)h_T[i + m - 1] * h_T[j + m - 1]
                    - (double)h_T[i - 1]      * h_T[j - 1];
            }

            // Compute mean/std for window i
            double si = 0, si2 = 0, sj = 0, sj2 = 0;
            for (int k = 0; k < m; k++) {
                double vi = h_T[i+k], vj = h_T[j+k];
                si += vi; si2 += vi*vi; sj += vj; sj2 += vj*vj;
            }
            double mu_i  = si/m,  mu_j  = sj/m;
            double sig_i = sqrt(fmax(0.0, si2/m - mu_i*mu_i)) + 1e-10;
            double sig_j = sqrt(fmax(0.0, sj2/m - mu_j*mu_j)) + 1e-10;
            double pear  = (qt/m - mu_i*mu_j) / (sig_i * sig_j);
            pear = fmin(1.0, fmax(-1.0, pear));
            float dist   = (float)sqrt(2.0 * m * (1.0 - pear));

            float tol = 1e-4f;
            if (fabsf(dist - h_mp[i]) < tol && h_mpi[i] == -1) h_mpi[i] = j;
            if (fabsf(dist - h_mp[j]) < tol && h_mpi[j] == -1) h_mpi[j] = i;
        }
    }
}

// -----------------------------------------------------------------------------
//  Host launcher
// -----------------------------------------------------------------------------
STOMPResult run_stomp(const float* h_T, int n, const STOMPConfig& cfg)
{
    STOMPResult result{};
    CudaTimer   gpu_timer;
    WallTimer   wall_timer;
    wall_timer.start();

    int m  = cfg.window_size;
    int ez = (int)floorf(cfg.ez_factor * (float)m);
    ez     = (ez < 1) ? 1 : ez;
    int L  = n - m + 1;

    if (L <= ez || L <= 0) {
        fprintf(stderr, "[stomp] Invalid config: L=%d ez=%d for m=%d n=%d\n",
                L, ez, m, n);
        return result;
    }

    // Upload T
    float* d_T;
    CUDA_CHECK(cudaMalloc(&d_T, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_T, h_T, n * sizeof(float), cudaMemcpyHostToDevice));

    // Pre-compute: stats and QT seeds
    float *d_means, *d_stds, *d_QT_init;
    gpu_timer.start();
    precompute(d_T, n, m, ez, &d_means, &d_stds, &d_QT_init);
    result.precompute_ms = gpu_timer.stop();

    // Allocate MP (init to +INF) and MPI (init to -1)
    float* d_MP;
    int*   d_MPI;
    CUDA_CHECK(cudaMalloc(&d_MP,  L * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_MPI, L * sizeof(int)));

    std::vector<float> init_mp(L, INF_F);
    CUDA_CHECK(cudaMemcpy(d_MP, init_mp.data(), L*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_MPI, -1, L * sizeof(int)));

    // Launch kernel
    gpu_timer.start();

    if (!cfg.approximate || cfg.approx_frac >= 0.999f) {
        // Full mode
        int n_diags = L - ez;
        dim3 block(BLOCK_SIZE);
        dim3 grid((n_diags + BLOCK_SIZE - 1) / BLOCK_SIZE);
        stomp_diagonal_kernel<<<grid, block>>>(
            d_T, d_means, d_stds, d_QT_init, d_MP, d_MPI, n, m, ez);
    } else {
        // Approximate mode: stride = 1/approx_frac
        int n_diags = L - ez;
        float frac  = fmaxf(0.05f, fminf(1.0f, cfg.approx_frac));
        int stride  = (int)ceilf(1.0f / frac);
        int n_active = (n_diags + stride - 1) / stride;

        dim3 block(BLOCK_SIZE);
        dim3 grid((n_active + BLOCK_SIZE - 1) / BLOCK_SIZE);
        stomp_approx_kernel<<<grid, block>>>(
            d_T, d_means, d_stds, d_QT_init, d_MP, d_MPI, n, m, ez, stride);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    result.kernel_ms = gpu_timer.stop();

    // Download
    result.profile_len = L;
    result.mp  = (float*)malloc(L * sizeof(float));
    result.mpi = (int*)  malloc(L * sizeof(int));
    CUDA_CHECK(cudaMemcpy(result.mp,  d_MP,  L*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result.mpi, d_MPI, L*sizeof(int),   cudaMemcpyDeviceToHost));

    // MPI recovery (optional CPU pass)
    if (cfg.recover_indices) {
        recover_mpi_cpu(h_T, result.mp, result.mpi, n, m, ez);
    }

    cudaFree(d_T); cudaFree(d_means); cudaFree(d_stds);
    cudaFree(d_QT_init); cudaFree(d_MP); cudaFree(d_MPI);

    result.total_ms = wall_timer.stop();
    return result;
}
