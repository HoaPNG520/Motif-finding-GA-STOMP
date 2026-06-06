#pragma once
// =============================================================================
//  benchmark.cuh -- Validation and reproducibility utilities
//
//  Two independent experiments:
//
//  1. run_multiseed
//     Runs the GA with N different random seeds on the same series.
//     Reports mean +/- std of: best_m, best_ez, best_fitness, wall_time.
//     Answers the conference question: "How reproducible are your results?"
//
//  2. run_fitness_recall_sweep
//     Sweeps m over a grid, computes both fitness score and motif recall
//     (fraction of top-k found motifs that overlap with the true injected
//     motif location).  Outputs a CSV for Spearman correlation analysis.
//     Answers: "Is your fitness function a valid proxy for motif quality?"
// =============================================================================

#include "common.cuh"
#include "ga.cuh"
#include "fitness.cuh"
#include <vector>

// -----------------------------------------------------------------------------
//  Multi-seed result
// -----------------------------------------------------------------------------
struct SeedResult
{
    unsigned int seed;
    int best_m;
    float best_ez;
    int best_k;
    float best_fitness;
    float wall_time_ms;
};

struct MultiseedReport
{
    std::vector<SeedResult> results;

    // Statistics computed by finalize()
    float mean_m, std_m;
    float mean_ez, std_ez;
    float mean_fit, std_fit;
    float mean_time, std_time;

    void finalize(); // compute mean/std from results
    void print() const;
    void save_csv(const char *path) const;
};

// Run the GA with each seed in seeds[0..n_seeds-1].
// Uses the same h_T and GAConfig for every run.
MultiseedReport run_multiseed(
    const float *h_T,
    int n,
    const GAConfig &cfg,
    const unsigned int *seeds,
    int n_seeds);

// -----------------------------------------------------------------------------
//  Fitness-recall sweep result
// -----------------------------------------------------------------------------
struct SweepRow
{
    int m;          // window size tested
    float fitness;  // composite fitness score
    float recall;   // fraction of true motif correctly identified
    float top_dist; // MP minimum (top motif distance)
    float autocorr_peak; // autocorrelation peak signal
    float contrast; // contrast signal
};

struct SweepReport
{
    std::vector<SweepRow> rows;
    float spearman_rho; // rank correlation between fitness and recall
    void print() const;
    void save_csv(const char *path) const;
};

// Sweep m from m_min to m_max in steps of step_size.
// true_pos1 / true_pos2 are the known injection positions (for recall calc).
// ez_factor and min_k are held fixed during the sweep.
SweepReport run_fitness_recall_sweep(
    const float *h_T,
    int n,
    int true_m,
    int true_pos1,
    int true_pos2,
    int m_min,
    int m_max,
    int step_size,
    float ez_factor = 0.25f,
    int min_k = 5);
