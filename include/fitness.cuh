#pragma once
// =============================================================================
//  fitness.cuh -- Unsupervised composite fitness for GA individuals
//
//  +----------------------+--------------------------------------------------+
//  |  Signal              |  Intuition                                       |
//  +----------------------+--------------------------------------------------+
//  |  Contrast  (w=0.40)  |  Gap between top motif and background.           |
//  |                      |  High gap -> motif is genuinely distinctive.     |
//  +----------------------+--------------------------------------------------+
//  |  Regularity(w=0.30)  |  Coefficient of Variation of motif gaps.         |
//  |                      |  High regularity -> motifs appear periodically.  |
//  +----------------------+--------------------------------------------------+
//  |  Consistency(w=0.20) |  Difference between mean motif gap and `m`.      |
//  |                      |  Ensures window size matches the discovered gap. |
//  +----------------------+--------------------------------------------------+
//  |  Count     (w=0.10)  |  Closeness of discovered motifs to target k.     |
//  +----------------------+--------------------------------------------------+
// =============================================================================

#include "common.cuh"
#include <vector>

struct Individual
{
    int window_size;     // m  in [m_min, m_max]
    float ez_factor;     // in [0.25, 1.0]
    int min_motif_count; // k  in [2, 20]
    float fitness;       // set by evaluate_fitness; -INF before evaluation
};

struct FitnessScore
{
    float composite;           // final weighted score in [0, 1]
    float contrast;            // signal 1
    float count_score;         // signal 2
    int discovered_motifs;     // how many motifs were found below threshold
    float spacing_regularity;  // NEW: Approach 6 regularity
    float spacing_consistency; // NEW: Approach 6 self-consistency
};

FitnessScore evaluate_fitness(
    const float *mp,
    int profile_len,
    const Individual &ind);

void print_fitness(const Individual &ind, const FitnessScore &fs);