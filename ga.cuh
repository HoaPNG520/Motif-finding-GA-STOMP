#pragma once
// =============================================================================
//  ga.cuh -- Genetic Algorithm for joint hyperparameter optimisation
//
//  Searches the 3D space [window_size x ez_factor x min_motif_count].
//
//  Operators:
//    Selection  -- k-tournament (k=3 default)
//    Crossover  -- arithmetic blend for floats/ints; inherit-one for discrete k
//    Mutation   -- Gaussian perturbation with temperature annealing
//    Elitism    -- top-2 individuals always survive
//
//  Each fitness evaluation invokes one full GPU-STOMP run (run_stomp).
//  Parallelism is WITHIN each evaluation (diagonal STOMP kernel), not across
//  the population -- this keeps GPU memory bounded to one MP array at a time.
// =============================================================================

#include "common.cuh"
#include "fitness.cuh"
#include <vector>

// -- GA hyperparameters (outer loop configuration) -----------------------------
struct GAConfig
{
    // Search bounds for Individual genes
    int m_min = 10;
    int m_max = 300;
    float ez_min = 0.25f;
    float ez_max = 1.0f;
    int k_min = 2;
    int k_max = 20;

    // GA parameters
    int population_size = 50;
    int generations = 30;
    int tournament_k = 3;
    float mutation_rate = 0.30f; // per-gene probability
    int elite_count = 2;         // individuals carried over unchanged

    // Logging
    bool verbose = true;
};

// -- GA output -----------------------------------------------------------------
struct GAResult
{
    Individual best_individual;
    FitnessScore best_fitness;
    std::vector<float> fitness_history; // best fitness per generation
};

// -- GA operators (declared here, defined in ga.cu) ---------------------------

// Uniformly random individual within bounds
Individual random_individual(const GAConfig &cfg, unsigned int seed);

// k-tournament selection (higher fitness wins)
Individual tournament_select(
    const std::vector<Individual> &pop,
    int k);

// Arithmetic crossover:
//   window_size      -> blend with alpha in [0.3, 0.7]
//   ez_factor        -> blend with same alpha
//   min_motif_count  -> inherit from one parent at random
Individual crossover(
    const Individual &a,
    const Individual &b,
    const GAConfig &cfg);

// Gaussian mutation with annealing:
//   temperature = 1 - (generation / max_generations)  in (0, 1]
//   mutation strength scales with temperature so search narrows over time
Individual mutate(
    Individual ind,
    float temperature,
    const GAConfig &cfg);

// -- Main GA loop --------------------------------------------------------------
// Drives the full evolution; calls run_stomp + evaluate_fitness per individual.
GAResult run_ga(
    const float *h_T, // [n] host time series
    int n,
    const GAConfig &cfg,
    unsigned int seed = 42);
