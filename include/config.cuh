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
//    sampling_rate;   // default 12000
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
    int sampling_rate;    // default 12000
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
