#pragma once
// =============================================================================
//  config.cuh -- Simple key=value configuration file parser
//
//  Format (plain text, one setting per line):
//    # comment lines start with #
//    m_min        = 10
//    m_max        = 300
//    ez_min       = 0.25
//    ez_max       = 1.0
//    k_min        = 2
//    k_max        = 20
//    population   = 50
//    generations  = 30
//    tournament_k = 3
//    mutation_rate= 0.30
//    elite_count  = 2
//    approx_frac  = 1.0    # 1.0 = full STOMP; 0.3 = 30% diagonals (faster)
//    n_seeds      = 5      # for multi-seed variance report
//    verbose      = 1
//    sampling_rate     = 12000  # Hz, 0 = unknown
//    fault_freq_min_hz = 50     # FFT search band in Hz (needs sampling_rate)
//    fault_freq_max_hz = 300
//    period_min   = 30     # fallback band in samples when Hz keys are unset
//    period_max   = 400
//    w_contrast   = 0.20   # fitness weights (set one to 0 for an ablation)
//    w_regularity = 0.30
//    w_period     = 0.50
//    period_bias_init = 1  # 0 = don't seed the population around the FFT period
//    landscape_step   = 0  # >0 writes landscape.csv: signals for every m step
//    null_correct     = 0  # 1 = contrast scored as real - surrogate
//    null_surrogate   = phase   # phase | shuffle
//    null_seed        = 12345
//  Missing keys fall back to GAConfig defaults.
//  Unknown keys are ignored with a warning.
// =============================================================================

#include "ga.cuh"

// Extend GAConfig with two extra fields used only via config file
struct FullConfig
{
    GAConfig ga;          // standard GA parameters
    float approx_frac;    // SCRIMP++ approximation fraction [0.1, 1.0]
    int n_seeds;          // number of random seeds for variance report
    char input_path[512]; // path to input CSV/bin (empty = synthetic)
    char output_dir[512]; // directory for output files
    // sampling_rate / fault band / period band / weights live in ga (GAConfig)
    int landscape_step;   // >0: after the GA, sweep m in steps of this and
                          // write landscape.csv (every fitness signal per m)
};

// Default values
FullConfig default_full_config();

// Parse a config file; missing keys use defaults.
// Returns false and prints an error if the file cannot be opened.
bool load_config(const char *path, FullConfig *out);

// Write the current config to a file (useful for reproducibility logging).
void save_config(const char *path, const FullConfig &cfg);

// Print all settings to stdout.
void print_config(const FullConfig &cfg);
