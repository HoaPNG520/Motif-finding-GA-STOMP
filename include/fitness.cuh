#pragma once
// =============================================================================
//  fitness.cuh -- Unsupervised composite fitness for GA individuals
//
//  Fitness is a weighted sum of three independent signals derived from the
//  Matrix Profile returned by STOMP.  No labels are required.
//
//  +----------------------+--------------------------------------------------+
//  |  Signal              |  Intuition                                       |
//  +----------------------+--------------------------------------------------+
//  |  Contrast  (w=0.40)  |  Gap between top motif and background.           |
//  |                      |  High gap -> motif is genuinely distinctive.     |
//  +----------------------+--------------------------------------------------+
//  |  Autocorr  (w=0.40)  |  Peak alignment with full MP autocorrelation.    |
//  |                      |  Rewards window sizes that match signal period.  |
//  +----------------------+--------------------------------------------------+
//  |  Count     (w=0.20)  |  Closeness of discovered motifs to target k.     |
//  |                      |  Penalises both too many and too few matches.    |
//  +----------------------+--------------------------------------------------+
// =============================================================================

#include "common.cuh"
#include <vector>

// -- GA individual: the three jointly-optimised hyperparameters ----------------
struct Individual
{
    int window_size;     // m  in [m_min, m_max]
    float ez_factor;     // in [0.25, 1.0]
    int min_motif_count; // k  in [2, 20]
    float fitness;       // set by evaluate_fitness; -INF before evaluation
};

static constexpr int HIST_BINS = 50; // retained for count_score threshold

// -- Decomposed fitness score (useful for logging and analysis) ----------------
struct FitnessScore
{
    float composite;       // final weighted score in [0, 1]
    float contrast;        // signal 1
    float autocorr_peak;   // signal 2: autocorrelation peak alignment
    float count_score;     // signal 3
    int discovered_motifs; // how many motifs were found below threshold
};

// -- Main fitness evaluator ----------------------------------------------------
// Runs entirely on the CPU using the host-side MP returned by run_stomp.
FitnessScore evaluate_fitness(
    const float *mp, // [profile_len]  Matrix Profile distances
    int profile_len,
    const Individual &ind);

// -- Pretty-print a fitness breakdown -----------------------------------------
void print_fitness(const Individual &ind, const FitnessScore &fs);