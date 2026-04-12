#pragma once
// =============================================================================
//  fitness.cuh — Unsupervised composite fitness for GA individuals
//
//  Fitness is a weighted sum of three independent signals derived from the
//  Matrix Profile returned by STOMP.  No labels are required.
//
//  ┌──────────────────────┬──────────────────────────────────────────────────┐
//  │  Signal              │  Intuition                                       │
//  ├──────────────────────┼──────────────────────────────────────────────────┤
//  │  Contrast  (w=0.50)  │  Gap between top motif and background.           │
//  │                      │  High gap → motif is genuinely distinctive.      │
//  ├──────────────────────┼──────────────────────────────────────────────────┤
//  │  Entropy   (w=0.30)  │  Histogram entropy of MP values.                 │
//  │                      │  Low entropy → trivial/degenerate profile (bad). │
//  ├──────────────────────┼──────────────────────────────────────────────────┤
//  │  Count     (w=0.20)  │  Closeness of discovered motifs to target k.    │
//  │                      │  Penalises both too many and too few matches.    │
//  └──────────────────────┴──────────────────────────────────────────────────┘
// =============================================================================

#include "common.cuh"
#include <vector>

// ── GA individual: the three jointly-optimised hyperparameters ────────────────
struct Individual {
    int   window_size;          // m  ∈ [m_min, m_max]
    float ez_factor;            // ∈ [0.25, 1.0]
    int   min_motif_count;      // k  ∈ [2, 20]
    float fitness;              // set by evaluate_fitness; -∞ before evaluation
};

// ── Decomposed fitness score (useful for logging and analysis) ────────────────
struct FitnessScore {
    float composite;            // final weighted score ∈ [0, 1]
    float contrast;             // signal 1
    float entropy;              // signal 2 (normalised)
    float count_score;          // signal 3
    int   discovered_motifs;    // how many motifs were found below threshold
};

// ── Fitness weights ───────────────────────────────────────────────────────────
static constexpr float W_CONTRAST = 0.50f;
static constexpr float W_ENTROPY  = 0.30f;
static constexpr float W_COUNT    = 0.20f;
static constexpr int   HIST_BINS  = 50;

// ── Main fitness evaluator ────────────────────────────────────────────────────
// Runs entirely on the CPU using the host-side MP returned by run_stomp.
FitnessScore evaluate_fitness(
    const float*      mp,   // [profile_len]  Matrix Profile distances
    int               profile_len,
    const Individual& ind
);

// ── Pretty-print a fitness breakdown ─────────────────────────────────────────
void print_fitness(const Individual& ind, const FitnessScore& fs);
