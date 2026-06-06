#pragma once
// =============================================================================
//  fitness.cuh -- Unsupervised composite fitness for GA individuals (v7 Combined)
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
    float autocorr_peak;       // signal 3 (from v3)
    float spacing_regularity;  // signal 4 (from v6)
    float spacing_consistency; // signal 5 (from v6)
};

FitnessScore evaluate_fitness(
    const float *mp,
    int profile_len,
    const Individual &ind);

void print_fitness(const Individual &ind, const FitnessScore &fs);