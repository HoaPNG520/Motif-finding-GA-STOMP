// =============================================================================
//  main.cu -- Driver for GA + GPU-STOMP
//
//  Usage:
//    ./ga_stomp [config.ini]
//    ./ga_stomp [n] [true_m] [pop] [gens]        (quick synthetic mode)
//
//  All outputs are written to output_dir (default: current directory).
//
//  Output files:
//    series.csv           -- synthetic time series (or copy of input)
//    matrix_profile.csv   -- final MP with distances and indices
//    timing.csv           -- kernel/GA/fitness timing breakdown
//    multiseed.csv        -- multi-seed reproducibility results
//    sweep.csv            -- fitness-recall sweep (Spearman correlation)
//    run_config.ini       -- exact config used (for reproducibility)
// =============================================================================

#include "common.cuh"
#include "stomp.cuh"
#include "fitness.cuh"
#include "ga.cuh"
#include "io.cuh"
#include "timer.cuh"
#include "config.cuh"
#include "benchmark.cuh"

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
    float* h_T, int n, int true_m, int pos1, int pos2,
    float sigma = 0.15f, unsigned int seed = 42)
{
    srand(seed);
    for (int i = 0; i < n; i++) h_T[i] = gaussianRand();

    std::vector<float> motif(true_m);
    for (int k = 0; k < true_m; k++)
        motif[k] = sinf(2.0f*(float)M_PI*(float)k/(float)true_m);

    for (int k = 0; k < true_m; k++)
        h_T[pos1+k] = motif[k] + sigma*gaussianRand();
    for (int k = 0; k < true_m; k++)
        h_T[pos2+k] = motif[k] + sigma*gaussianRand();
}

// -----------------------------------------------------------------------------
//  Find top motif index
// -----------------------------------------------------------------------------
static int find_top_motif(const float* mp, int L)
{
    int best = 0;
    for (int i = 1; i < L; i++)
        if (mp[i] < mp[best]) best = i;
    return best;
}

// -----------------------------------------------------------------------------
//  ASCII Matrix Profile visualiser
//  Characters represent MP value from low to high: . : i I H #
// -----------------------------------------------------------------------------
static void print_mp_ascii(const float* mp, int L, int width=60)
{
    float lo = *std::min_element(mp, mp+L);
    float hi = *std::max_element(mp, mp+L);
    float range = hi - lo + EPS;
    const char bars[] = {'.', ':', 'i', 'I', 'H', '#'};
    int nb = (int)(sizeof(bars)/sizeof(bars[0]));

    printf("\n  Matrix Profile (downsampled to %d cols)\n", width);
    printf("  Min=%.3f  [", lo);
    for (int c = 0; c < width; c++) {
        int idx = (int)((float)c/(float)width*(float)L);
        idx = std::min(idx, L-1);
        int b = (int)((float)(nb-1)*(mp[idx]-lo)/range);
        printf("%c", bars[std::max(0,std::min(b,nb-1))]);
    }
    printf("]  Max=%.3f\n\n", hi);
}

// -----------------------------------------------------------------------------
//  ASCII convergence chart
// -----------------------------------------------------------------------------
static void print_convergence(const std::vector<float>& hist)
{
    int n = (int)hist.size();
    float lo = *std::min_element(hist.begin(), hist.end());
    float hi = *std::max_element(hist.begin(), hist.end());
    float range = hi - lo + EPS;
    int rows = 8;

    printf("\n  Fitness convergence\n");
    for (int row = rows; row >= 0; row--) {
        float thr = lo + (float)row/(float)rows*range;
        printf("  %5.3f | ", thr);
        for (int g = 0; g < n; g++)
            printf("%c", hist[g] >= thr ? '#' : ' ');
        printf("\n");
    }
    printf("         +-");
    for (int g = 0; g < n; g++) printf("-");
    printf("\n          Gen0");
    for (int i = 0; i < n-8; i++) printf(" ");
    printf("Gen%d\n\n", n-1);
}

// -----------------------------------------------------------------------------
//  GPU device info
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
//  Build output path:  output_dir + "/" + filename
// -----------------------------------------------------------------------------
static void out_path(char* buf, int bufsz,
                     const char* dir, const char* filename)
{
    // Ensure trailing slash on dir
    int dlen = (int)strlen(dir);
    if (dlen > 0 && dir[dlen-1] == '/')
        snprintf(buf, bufsz, "%s%s", dir, filename);
    else
        snprintf(buf, bufsz, "%s/%s", dir, filename);
}

// -----------------------------------------------------------------------------
//  Print help
// -----------------------------------------------------------------------------
static void print_help(const char* argv0)
{
    printf("Usage:\n");
    printf("  %s                         -- synthetic run with defaults\n", argv0);
    printf("  %s config.ini              -- load settings from file\n", argv0);
    printf("  %s [n] [true_m] [pop] [gens] -- quick synthetic mode\n", argv0);
    printf("  %s --help                  -- show this message\n\n", argv0);
    printf("Config file keys (with defaults):\n");
    printf("  m_min=10  m_max=300  ez_min=0.25  ez_max=1.0\n");
    printf("  k_min=2  k_max=20  population=50  generations=30\n");
    printf("  tournament_k=3  mutation_rate=0.30  elite_count=2\n");
    printf("  approx_frac=1.0   # <1.0 enables SCRIMP++ approximation\n");
    printf("  n_seeds=5         # for multi-seed reproducibility report\n");
    printf("  verbose=1\n");
    printf("  input_path=       # blank = use synthetic series\n");
    printf("  output_dir=.      # directory for all output CSV files\n");
}

// =============================================================================
//  Entry point
// =============================================================================
int main(int argc, char** argv)
{
    // -- Parse arguments -------------------------------------------------------
    if (argc >= 2 && (!strcmp(argv[1], "--help") || !strcmp(argv[1], "-h"))) {
        print_help(argv[0]); return 0;
    }

    FullConfig fc = default_full_config();
    int  true_m  = 80;
    int  true_n  = 4000;
    int  true_pos1 = 200;
    int  true_pos2 = -1;   // set below
    bool use_config_file = false;

    if (argc == 2) {
        // Check if the argument looks like a config file
        const char* arg = argv[1];
        int alen = (int)strlen(arg);
        if (alen > 4 && (!strcmp(arg+alen-4, ".ini") || !strcmp(arg+alen-4, ".cfg"))) {
            if (!load_config(arg, &fc)) return 1;
            use_config_file = true;
        } else {
            true_n = atoi(arg);
        }
    } else if (argc >= 3) {
        true_n  = atoi(argv[1]);
        true_m  = atoi(argv[2]);
        if (argc >= 4) fc.ga.population_size = atoi(argv[3]);
        if (argc >= 5) fc.ga.generations     = atoi(argv[4]);
    }

    if (true_n < 200)       { fprintf(stderr, "n must be >= 200\n"); return 1; }
    if (true_m < 4)         { fprintf(stderr, "true_m must be >= 4\n"); return 1; }
    if (true_m * 3 > true_n){ fprintf(stderr, "true_m too large\n"); return 1; }

    true_pos2 = true_n / 2 + 100;

    // -- Print banner ----------------------------------------------------------
    print_device_info();
    if (!use_config_file) {
        // Set sensible GA bounds around true_m
        fc.ga.m_min = std::max(4,       true_m/3);
        fc.ga.m_max = std::min(true_n/4, true_m*3);
    }
    print_config(fc);

    // -- Load or generate time series ------------------------------------------
    float* h_T  = nullptr;
    int    n_ts = 0;

    if (fc.input_path[0] != '\0') {
        // Determine format by extension
        const char* ext = strrchr(fc.input_path, '.');
        if (ext && !strcmp(ext, ".bin"))
            h_T = load_series_bin(fc.input_path, &n_ts);
        else
            h_T = load_series_csv(fc.input_path, &n_ts);

        if (!h_T) return 1;
        true_m   = 0;   // unknown for real data
        true_pos1 = -1;
        true_pos2 = -1;
    } else {
        n_ts = true_n;
        h_T  = (float*)malloc(n_ts * sizeof(float));
        generate_synthetic(h_T, n_ts, true_m, true_pos1, true_pos2);
        printf("Synthetic series: n=%d  true_m=%d  motif at [%d, %d]\n\n",
               n_ts, true_m, true_pos1, true_pos2);
    }

    // -- Save a copy of the series for reference -------------------------------
    char tmp_path[600];
    out_path(tmp_path, sizeof(tmp_path), fc.output_dir, "series.csv");
    save_series_csv(tmp_path, h_T, n_ts);

    // -- Save the config used for this run ------------------------------------
    out_path(tmp_path, sizeof(tmp_path), fc.output_dir, "run_config.ini");
    save_config(tmp_path, fc);

    // -- Run GA ----------------------------------------------------------------
    GAResult ga_result = run_ga(h_T, n_ts, fc.ga, /*seed=*/42);

    print_convergence(ga_result.fitness_history);

    // -- Save timing -----------------------------------------------------------
    out_path(tmp_path, sizeof(tmp_path), fc.output_dir, "timing.csv");
    ga_result.timing.save_csv(tmp_path);

    // -- Final STOMP with MPI recovery ----------------------------------------
    printf("Running final STOMP (full mode, MPI recovery enabled)...\n");
    STOMPConfig best_cfg;
    best_cfg.window_size     = ga_result.best_individual.window_size;
    best_cfg.ez_factor       = ga_result.best_individual.ez_factor;
    best_cfg.min_motif_count = ga_result.best_individual.min_motif_count;
    best_cfg.approximate     = false;
    best_cfg.approx_frac     = 1.0f;
    best_cfg.recover_indices = true;

    STOMPResult final_res = run_stomp(h_T, n_ts, best_cfg);

    // -- Save MP ---------------------------------------------------------------
    out_path(tmp_path, sizeof(tmp_path), fc.output_dir, "matrix_profile.csv");
    save_mp_csv(tmp_path, final_res.mp, final_res.mpi,
                final_res.profile_len, best_cfg.window_size);

    // -- Visualise MP ----------------------------------------------------------
    print_mp_ascii(final_res.mp, final_res.profile_len);

    // -- Report top motif ------------------------------------------------------
    int top = find_top_motif(final_res.mp, final_res.profile_len);
    printf("+--------------------------------------------------------------+\n");
    printf("|  Top motif index     : %d\n", top);
    if (true_pos1 >= 0) {
        printf("|  True motif pos1     : %d\n", true_pos1);
        printf("|  Location error      : %d positions\n", abs(top - true_pos1));
        printf("|  Window size error   : %d\n",
               abs(ga_result.best_individual.window_size - true_m));
    }
    printf("+--------------------------------------------------------------+\n\n");

    // -- Multi-seed reproducibility report ------------------------------------
    printf("Running multi-seed reproducibility experiment (%d seeds)...\n",
           fc.n_seeds);
    // Use deterministic seeds derived from base seed 42
    std::vector<unsigned int> seeds(fc.n_seeds);
    for (int s = 0; s < fc.n_seeds; s++) seeds[s] = 42 + (unsigned)s * 1000;

    // Quiet mode for multi-seed (avoid wall of text)
    GAConfig quiet_cfg = fc.ga;
    quiet_cfg.verbose = false;

    MultiseedReport ms_report = run_multiseed(
        h_T, n_ts, quiet_cfg, seeds.data(), fc.n_seeds);
    ms_report.print();

    out_path(tmp_path, sizeof(tmp_path), fc.output_dir, "multiseed.csv");
    ms_report.save_csv(tmp_path);

    // -- Fitness-recall sweep (only for synthetic series where truth is known) --
    if (true_pos1 >= 0 && true_m > 0) {
        printf("Running fitness-recall sweep for Spearman validation...\n");
        int step = std::max(1, (fc.ga.m_max - fc.ga.m_min) / 20);
        SweepReport sw = run_fitness_recall_sweep(
            h_T, n_ts, true_m, true_pos1, true_pos2,
            fc.ga.m_min, fc.ga.m_max, step,
            /*ez_factor=*/0.25f, /*min_k=*/5);
        sw.print();

        out_path(tmp_path, sizeof(tmp_path), fc.output_dir, "sweep.csv");
        sw.save_csv(tmp_path);
    } else {
        printf("[main] Fitness-recall sweep skipped (real data -- no ground truth)\n\n");
    }

    // -- Cleanup ---------------------------------------------------------------
    free(final_res.mp);
    free(final_res.mpi);
    free(h_T);

    printf("All output files written to: %s/\n", fc.output_dir);
    return 0;
}
