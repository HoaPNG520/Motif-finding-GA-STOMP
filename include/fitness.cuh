#pragma once
// =============================================================================
//  fitness.cuh  --  Unsupervised composite fitness for GA-STOMP individuals
//
//  Two signals combined as a weighted sum:
//
//  +---------------------+----------------------------------------------------+
//  |  Signal             |  Intuition                                         |
//  +---------------------+----------------------------------------------------+
//  |  Contrast  (w=0.35) |  Gap between top-motif distance and background.    |
//  |                     |  High gap → motif is genuinely distinctive.        |
//  +---------------------+----------------------------------------------------+
//  |  Regularity(w=0.65) |  Coefficient of Variation of gaps between the      |
//  |                     |  FIXED_COUNT lowest-MP positions.                  |
//  |                     |  Low CV → motifs recur at a consistent period      |
//  |                     |  → window aligns with the true defect frequency.   |
//  +---------------------+----------------------------------------------------+
//
//  Design rationale for fixed-count selection:
//    A percentage-based threshold (e.g. top 5 %) always selects
//    profile_len/20 positions, giving mean_gap ≈ 20 regardless of window
//    size.  This makes the regularity signal window-size-blind and creates
//    a systematic bias toward small windows.  Using a fixed count of 50
//    decouples mean_gap from window size (mean_gap ≈ profile_len/50 ≈ 118),
//    so the CV of gaps purely measures periodic recurrence quality.
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
    float composite;           // final weighted score ∈ [0, 1]
    float contrast;            // Signal 1: motif distinctiveness
    float spacing_regularity;  // Signal 2: periodic recurrence of top-N motifs
    float count_score;         // legacy field (computed but not used in composite)
    int   discovered_motifs;   // number of subsequences below count threshold
    float spacing_consistency; // legacy field (mirrors regularity for logging)
};

FitnessScore evaluate_fitness(
    const float *mp,
    int          profile_len,
    const Individual &ind);

void print_fitness(const Individual &ind, const FitnessScore &fs);
