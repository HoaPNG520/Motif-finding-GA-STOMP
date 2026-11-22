#pragma once
// =============================================================================
//  ga.cuh -- Genetic Algorithm for joint hyperparameter optimisation
//
//  Searches the 3D space [window_size x ez_factor x min_motif_count].
//
//  Operators:
//    Selection  -- k-tournament (k=3 default)
//    Crossover  -- arithmetic blend (alpha in [0.3,0.7]) for continuous genes;
//                  random-parent inheritance for discrete min_motif_count
//    Mutation   -- Gaussian perturbation with temperature annealing
//    Elitism    -- top-2 individuals always survive
// =============================================================================

#include "common.cuh"
#include "fitness.cuh"
#include "timer.cuh"
#include <vector>

// -----------------------------------------------------------------------------
//  GA hyperparameters
// -----------------------------------------------------------------------------
struct GAConfig
{
    int m_min = 10;
    int m_max = 300;
    float ez_min = 0.25f;
    float ez_max = 1.0f;
    int k_min = 2;
    int k_max = 20;

    int population_size = 50;
    int generations = 30;
    int tournament_k = 3;
    float mutation_rate = 0.30f;
    int elite_count = 2;

    // SCRIMP++ approximation: during GA search use approx_frac < 1.0
    // to evaluate more individuals per second. Final run always uses 1.0.
    float approx_frac = 1.0f;

    // FFT reference-period search band.
    // Physical mode (preferred for real sensors): set sampling_rate (Hz) and
    // the expected fault/event frequency band in Hz.  The band in samples is
    //   [ sampling_rate / fault_freq_max_hz , sampling_rate / fault_freq_min_hz ]
    // e.g. CWRU 12 kHz, BPFI 162 Hz -> 74 samples, BPFO 107 Hz -> 112 samples.
    // Sample mode (fallback): period_min / period_max directly in samples.
    int   sampling_rate     = 0;     // Hz; 0 = unknown
    float fault_freq_min_hz = 0.0f;  // Hz; 0 = unset
    float fault_freq_max_hz = 0.0f;  // Hz; 0 = unset
    int   period_min        = 30;    // samples (fallback)
    int   period_max        = 400;   // samples (fallback)

    bool verbose = true;
};

// -----------------------------------------------------------------------------
//  GA output
// -----------------------------------------------------------------------------
struct GAResult
{
    Individual best_individual;
    FitnessScore best_fitness;
    std::vector<float> fitness_history; // best fitness per generation
    std::vector<float> mean_fitness_history; // population mean per generation
    int detected_period = -1;           // FFT consensus period (samples)
    int period_band_min = -1;           // band actually searched (samples)
    int period_band_max = -1;
    TimingReport timing;
};

// -----------------------------------------------------------------------------
//  GA operators
// -----------------------------------------------------------------------------
Individual random_individual(const GAConfig &cfg, unsigned int seed, int detected_period = -1);

// Resolve the FFT period search band [*pmin, *pmax] in samples.
// Uses sampling_rate + fault_freq_{min,max}_hz when all three are set,
// otherwise falls back to period_min / period_max.  Clamped to [2, n/2].
void resolve_period_band(const GAConfig &cfg, int n, int *pmin, int *pmax);

Individual tournament_select(
    const std::vector<Individual> &pop,
    int k);

Individual crossover(
    const Individual &a,
    const Individual &b,
    const GAConfig &cfg);

Individual mutate(
    Individual ind,
    float temperature,
    const GAConfig &cfg);

// -----------------------------------------------------------------------------
//  Main GA loop
// -----------------------------------------------------------------------------
GAResult run_ga(
    const float *h_T,
    int n,
    const GAConfig &cfg,
    unsigned int seed = 42);
