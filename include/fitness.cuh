#pragma once
// =============================================================================
//  fitness.cuh  --  Unsupervised composite fitness for GA-STOMP individuals
//
//  Three signals combined as a weighted sum:
//
//  +---------------------+----------------------------------------------------+
//  |  Signal             |  Intuition                                         |
//  +---------------------+----------------------------------------------------+
//  |  Contrast  (w=0.20) |  Gap between top-motif distance and background.    |
//  |                     |  High gap → motif is genuinely distinctive.        |
//  +---------------------+----------------------------------------------------+
//  |  Regularity(w=0.30) |  Distinct motif instances (>= m apart); how well   |
//  |                     |  their gaps fit integer multiples of the median    |
//  |                     |  gap.  1 = periodic, 0 = random spacing.           |
//  +---------------------+----------------------------------------------------+
//  |  Period Reward(w=0.50)| Gaussian reward centered on FFT-detected period.  |
//  |                     |  Strongly guides GA toward true defect frequency.  |
//  +---------------------+----------------------------------------------------+
//
//  Both contrast and regularity work on DISTINCT motif instances picked by
//  non-maximum suppression (positions >= m apart).  Raw "k-th smallest MP" or
//  "top-50 positions" are dominated by trivial neighbours of one occurrence.
// =============================================================================

#include "common.cuh"
#include <vector>

struct Individual
{
    int   window_size;      // m  ∈ [m_min, m_max]
    float ez_factor;        // exclusion zone ∈ [0.25, 1.0]
    int   min_motif_count;  // k  ∈ [2, 20]
    float fitness;          // set by evaluate_fitness; -INF before evaluation
};

struct FitnessScore
{
    float composite;            // final weighted score ∈ [0, 1]
    float contrast;             // Signal 1: motif distinctiveness
    float spacing_regularity;   // Signal 2: periodic recurrence of top-N motifs
    float period_reward;        // Signal 3: Gaussian reward around detected period
    float count_score;          // legacy field (computed but not used in composite)
    int   discovered_motifs;    // number of subsequences below count threshold
    float spacing_consistency;  // legacy field (mirrors regularity for logging)
};

// Composite weights (defaults match the v6 design; override per run for ablations)
struct FitnessWeights
{
    float contrast   = 0.20f;
    float regularity = 0.30f;
    float period     = 0.50f;
};

// Detect dominant period from MP using FFT (returns period in samples, or -1)
int detect_dominant_period(const float *mp, int profile_len, int min_period, int max_period);

// Gaussian reward centered on detected period
float compute_period_reward(int window_size, int detected_period, float sigma);

// Main fitness evaluator (with optional period reward)
FitnessScore evaluate_fitness(
    const float          *mp,
    int                   profile_len,
    const Individual     &ind,
    int                   detected_period = -1,          // -1 = no period reward
    const FitnessWeights &w = FitnessWeights());

void print_fitness(const Individual &ind, const FitnessScore &fs);
