// =============================================================================
//  main.cu -- Driver for GA + GPU-STOMP hyperparameter optimisation
//
//  Workflow:
//    1. Generate a synthetic time series with two injected motifs
//    2. Run the GA to discover the optimal [m, ez_factor, k]
//    3. Run STOMP with the best individual and report top motif locations
//    4. Compare discovered m against the true injected motif length
//
//  Usage:
//    ./ga_stomp [series_length] [true_motif_length] [population] [generations]
//
//  Example:
//    ./ga_stomp 5000 80 50 25
// =============================================================================

#include "common.cuh"
#include "stomp.cuh"
#include "fitness.cuh"
#include "ga.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

// -----------------------------------------------------------------------------
//  Synthetic time series generator
// -----------------------------------------------------------------------------
static void generate_synthetic(
    float *h_T,
    int n,
    int true_m,
    int pos1,
    int pos2,
    float motif_noise_sigma = 0.15f,
    unsigned int seed = 42)
{
    srand(seed);

    for (int i = 0; i < n; i++)
        h_T[i] = gaussianRand();

    std::vector<float> motif(true_m);
    for (int k = 0; k < true_m; k++)
    {
        motif[k] = sinf(2.0f * (float)M_PI * (float)k / (float)true_m);
    }

    for (int k = 0; k < true_m; k++)
        h_T[pos1 + k] = motif[k] + motif_noise_sigma * gaussianRand();

    for (int k = 0; k < true_m; k++)
        h_T[pos2 + k] = motif[k] + motif_noise_sigma * gaussianRand();
}

// -----------------------------------------------------------------------------
//  Find top motif pair from a Matrix Profile
// -----------------------------------------------------------------------------
static int find_top_motif(const float *mp, int profile_len)
{
    int best_i = 0;
    float best_val = mp[0];
    for (int i = 1; i < profile_len; i++)
    {
        if (mp[i] < best_val)
        {
            best_val = mp[i];
            best_i = i;
        }
    }
    return best_i;
}

// -----------------------------------------------------------------------------
//  Print Matrix Profile as ASCII bar chart
//  Levels (low to high): . : i I H #
// -----------------------------------------------------------------------------
static void print_mp_ascii(const float *mp, int profile_len, int width = 60)
{
    float mp_min = *std::min_element(mp, mp + profile_len);
    float mp_max = *std::max_element(mp, mp + profile_len);
    float range = mp_max - mp_min + EPS;

    const char bars[] = {'.', ':', 'i', 'I', 'H', '#'};
    int nb = (int)(sizeof(bars) / sizeof(bars[0]));

    printf("\n  Matrix Profile (downsampled to %d columns)\n", width);
    printf("  Max=%.3f  [", mp_max);
    for (int col = 0; col < width; col++)
    {
        int idx = (int)((float)col / (float)width * (float)profile_len);
        idx = std::min(idx, profile_len - 1);
        int b = (int)((float)(nb - 1) * (mp[idx] - mp_min) / range);
        b = std::max(0, std::min(b, nb - 1));
        printf("%c", bars[b]);
    }
    printf("]  Min=%.3f\n\n", mp_min);
}

// -----------------------------------------------------------------------------
//  Print fitness convergence curve (ASCII grid)
// -----------------------------------------------------------------------------
static void print_convergence(const std::vector<float> &history)
{
    int n = (int)history.size();
    float f_min = *std::min_element(history.begin(), history.end());
    float f_max = *std::max_element(history.begin(), history.end());
    float range = f_max - f_min + EPS;
    int rows = 8;

    printf("\n  Fitness convergence (each column = one generation)\n");
    for (int row = rows; row >= 0; row--)
    {
        float threshold = f_min + (float)row / (float)rows * range;
        printf("  %5.3f | ", threshold);
        for (int g = 0; g < n; g++)
            printf("%c", history[g] >= threshold ? '#' : ' ');
        printf("\n");
    }
    printf("         +-");
    for (int g = 0; g < n; g++)
        printf("-");
    printf("\n          Gen0");
    for (int i = 0; i < n - 8; i++)
        printf(" ");
    printf("Gen%d\n\n", n - 1);
}

// -----------------------------------------------------------------------------
//  Print GPU device info
// -----------------------------------------------------------------------------
static void print_device_info()
{
    int dev = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("+--------------------------------------------------------------+\n");
    printf("|  GPU: %-54s|\n", prop.name);
    printf("|  SMs: %-3d  | Global mem: %-5lu MB  | MaxThreads/Blk: %-5d |\n",
           prop.multiProcessorCount,
           (unsigned long)(prop.totalGlobalMem >> 20),
           prop.maxThreadsPerBlock);
    printf("|  Shared mem/SM: %-5lu KB  | Warp size: %-3d                 |\n",
           (unsigned long)(prop.sharedMemPerMultiprocessor >> 10),
           prop.warpSize);
    printf("+--------------------------------------------------------------+\n\n");
}

// -----------------------------------------------------------------------------
//  Entry point
// -----------------------------------------------------------------------------
int main(int argc, char **argv)
{
    int n = (argc > 1) ? atoi(argv[1]) : 4000;
    int true_m = (argc > 2) ? atoi(argv[2]) : 80;
    int pop_size = (argc > 3) ? atoi(argv[3]) : 30;
    int gens = (argc > 4) ? atoi(argv[4]) : 20;

    if (n < 200)
    {
        fprintf(stderr, "n must be >= 200\n");
        return 1;
    }
    if (true_m < 4)
    {
        fprintf(stderr, "true_m must be >= 4\n");
        return 1;
    }
    if (true_m * 3 > n)
    {
        fprintf(stderr, "true_m too large for n\n");
        return 1;
    }

    print_device_info();

    int pos1 = 200;
    int pos2 = n / 2 + 100;

    std::vector<float> h_T(n);
    generate_synthetic(h_T.data(), n, true_m, pos1, pos2);

    printf("Synthetic series:  n=%d  |  true_m=%d  |  motif at [%d, %d]\n\n",
           n, true_m, pos1, pos2);

    GAConfig cfg;
    cfg.m_min = std::max(4, true_m / 3);
    cfg.m_max = std::min(n / 4, true_m * 3);
    cfg.ez_min = 0.25f;
    cfg.ez_max = 1.0f;
    cfg.k_min = 2;
    cfg.k_max = 20;
    cfg.population_size = pop_size;
    cfg.generations = gens;
    cfg.tournament_k = 3;
    cfg.mutation_rate = 0.30f;
    cfg.elite_count = 2;
    cfg.verbose = true;

    GAResult ga_result = run_ga(h_T.data(), n, cfg, /*seed=*/42);

    printf("Running final STOMP with best hyperparameters...\n");
    STOMPConfig best_stomp_cfg;
    best_stomp_cfg.window_size = ga_result.best_individual.window_size;
    best_stomp_cfg.ez_factor = ga_result.best_individual.ez_factor;
    best_stomp_cfg.min_motif_count = ga_result.best_individual.min_motif_count;

    STOMPResult final_res = run_stomp(h_T.data(), n, best_stomp_cfg);

    int top_motif_idx = find_top_motif(final_res.mp, final_res.profile_len);

    printf("+--------------------------------------------------------------+\n");
    printf("|  Top motif location  : index %5d                           |\n", top_motif_idx);
    printf("|  True motif location : index %5d  (pos1=%d)               |\n", pos1, pos1);
    printf("|  Discovered m        : %-5d                                 |\n",
           ga_result.best_individual.window_size);
    printf("|  True m              : %-5d                                 |\n", true_m);
    printf("|  Location error      : %-5d positions                       |\n",
           abs(top_motif_idx - pos1));
    printf("|  Window size error   : %-5d                                 |\n",
           abs(ga_result.best_individual.window_size - true_m));
    printf("+--------------------------------------------------------------+\n");

    print_mp_ascii(final_res.mp, final_res.profile_len);
    print_convergence(ga_result.fitness_history);

    free(final_res.mp);
    free(final_res.mpi);

    return 0;
}
