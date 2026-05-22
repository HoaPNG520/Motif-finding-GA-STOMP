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
struct GAConfig {
    int   m_min            = 10;
    int   m_max            = 300;
    float ez_min           = 0.25f;
    float ez_max           = 1.0f;
    int   k_min            = 2;
    int   k_max            = 20;

    int   population_size  = 50;
    int   generations      = 30;
    int   tournament_k     = 3;
    float mutation_rate    = 0.30f;
    int   elite_count      = 2;

    // SCRIMP++ approximation: during GA search use approx_frac < 1.0
    // to evaluate more individuals per second. Final run always uses 1.0.
    float approx_frac      = 1.0f;

    bool  verbose          = true;
};

// -----------------------------------------------------------------------------
//  GA output
// -----------------------------------------------------------------------------
struct GAResult {
    Individual    best_individual;
    FitnessScore  best_fitness;
    std::vector<float> fitness_history;   // best fitness per generation
    TimingReport  timing;
};

// -----------------------------------------------------------------------------
//  GA operators
// -----------------------------------------------------------------------------
Individual random_individual(const GAConfig& cfg, unsigned int seed);

Individual tournament_select(
    const std::vector<Individual>& pop,
    int k
);

Individual crossover(
    const Individual& a,
    const Individual& b,
    const GAConfig&   cfg
);

Individual mutate(
    Individual      ind,
    float           temperature,
    const GAConfig& cfg
);

// -----------------------------------------------------------------------------
//  Main GA loop
// -----------------------------------------------------------------------------
GAResult run_ga(
    const float*    h_T,
    int             n,
    const GAConfig& cfg,
    unsigned int    seed = 42
);
