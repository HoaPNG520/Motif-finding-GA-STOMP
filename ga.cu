// =============================================================================
//  ga.cu — Genetic Algorithm for [window_size × ez_factor × min_motif_count]
//
//  Each fitness evaluation = one full GPU-STOMP run on the target series.
//  The GA outer loop runs on the CPU; all inner parallelism is within STOMP.
//
//  Operator summary:
//    Initialise : uniform random within bounds per gene
//    Select     : k-tournament (default k=3), deterministic given seed
//    Crossover  : arithmetic blend (α ∈ [0.3,0.7]) for continuous genes;
//                 random-parent inheritance for discrete min_motif_count
//    Mutate     : Gaussian(0, σ·temp) additive noise, σ = 10% of range;
//                 temperature anneals linearly: temp = 1 - gen/max_gen
//    Elitism    : top-2 survive unchanged every generation
// =============================================================================

#include "ga.cuh"
#include "stomp.cuh"
#include "fitness.cuh"
#include "common.cuh"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>
#include <numeric>

// ─────────────────────────────────────────────────────────────────────────────
//  Random individual — uniform within bounds
// ─────────────────────────────────────────────────────────────────────────────
Individual random_individual(const GAConfig& cfg, unsigned int seed)
{
    srand(seed);
    Individual ind;
    ind.window_size      = cfg.m_min + rand() % (cfg.m_max - cfg.m_min + 1);
    ind.ez_factor        = cfg.ez_min +
        (cfg.ez_max - cfg.ez_min) * ((float)rand() / (float)RAND_MAX);
    ind.min_motif_count  = cfg.k_min  + rand() % (cfg.k_max - cfg.k_min + 1);
    ind.fitness          = -1.0f;
    return ind;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Tournament selection
// ─────────────────────────────────────────────────────────────────────────────
Individual tournament_select(
    const std::vector<Individual>& pop,
    int k)
{
    Individual best = pop[rand() % pop.size()];
    for (int i = 1; i < k; i++) {
        Individual candidate = pop[rand() % pop.size()];
        if (candidate.fitness > best.fitness) best = candidate;
    }
    return best;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Arithmetic crossover
// ─────────────────────────────────────────────────────────────────────────────
//  alpha ∈ [0.3, 0.7] ensures the child is always between the two parents,
//  preventing rapid drift to either extreme.
//
//  window_size : blend → round to nearest integer
//  ez_factor   : blend (continuous)
//  min_motif_k : inherit from one parent at random (blending 7 and 3 → 5
//                is rarely meaningful for a count parameter)
Individual crossover(
    const Individual& a,
    const Individual& b,
    const GAConfig& cfg)
{
    float alpha = 0.3f + 0.4f * ((float)rand() / (float)RAND_MAX);
    Individual child;

    // Window size — arithmetic blend with integer rounding
    float blended_m = alpha * (float)a.window_size
                    + (1.0f - alpha) * (float)b.window_size;
    child.window_size = clampVal((int)roundf(blended_m), cfg.m_min, cfg.m_max);

    // Exclusion zone factor — arithmetic blend
    float blended_ez = alpha * a.ez_factor
                     + (1.0f - alpha) * b.ez_factor;
    child.ez_factor = clampVal(blended_ez, cfg.ez_min, cfg.ez_max);

    // Motif count — random inheritance (discrete, blending is meaningless)
    child.min_motif_count = (rand() % 2 == 0)
                            ? a.min_motif_count
                            : b.min_motif_count;

    child.fitness = -1.0f;
    return child;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Adaptive Gaussian mutation with temperature annealing
// ─────────────────────────────────────────────────────────────────────────────
//  temperature ∈ (0, 1] decreases each generation so the search narrows.
//  σ_m  = (m_max  - m_min)  * 0.10 * temperature
//  σ_ez = (ez_max - ez_min) * 0.10 * temperature
Individual mutate(
    Individual      ind,
    float           temperature,
    const GAConfig& cfg)
{
    // Window size
    if ((float)rand() / RAND_MAX < cfg.mutation_rate) {
        float sigma = (float)(cfg.m_max - cfg.m_min) * 0.10f * temperature;
        int   delta = (int)roundf(gaussianRand() * sigma);
        ind.window_size = clampVal(ind.window_size + delta,
                                   cfg.m_min, cfg.m_max);
    }

    // Exclusion zone factor
    if ((float)rand() / RAND_MAX < cfg.mutation_rate) {
        float sigma = (cfg.ez_max - cfg.ez_min) * 0.10f * temperature;
        float delta = gaussianRand() * sigma;
        ind.ez_factor = clampVal(ind.ez_factor + delta,
                                  cfg.ez_min, cfg.ez_max);
    }

    // Motif count — discrete ±1 mutation
    if ((float)rand() / RAND_MAX < cfg.mutation_rate * 0.5f) {
        int delta = (rand() % 3) - 1;      // -1, 0, or +1
        ind.min_motif_count = clampVal(ind.min_motif_count + delta,
                                        cfg.k_min, cfg.k_max);
    }

    ind.fitness = -1.0f;
    return ind;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Evaluate one individual — runs STOMP on GPU, computes fitness on CPU
// ─────────────────────────────────────────────────────────────────────────────
static FitnessScore evaluate_individual(
    Individual&    ind,
    const float*   h_T,
    int            n)
{
    STOMPConfig cfg;
    cfg.window_size     = ind.window_size;
    cfg.ez_factor       = ind.ez_factor;
    cfg.min_motif_count = ind.min_motif_count;

    STOMPResult res = run_stomp(h_T, n, cfg);

    FitnessScore fs{};
    if (res.mp != nullptr) {
        fs = evaluate_fitness(res.mp, res.profile_len, ind);
        free(res.mp);
        free(res.mpi);
    }

    ind.fitness = fs.composite;
    return fs;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Main GA loop
// ─────────────────────────────────────────────────────────────────────────────
GAResult run_ga(
    const float*    h_T,
    int             n,
    const GAConfig& cfg,
    unsigned int    seed)
{
    srand(seed);

    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║         GA + STOMP  Hyperparameter Optimisation             ║\n");
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║  Series length  : %-5d                                     ║\n", n);
    printf("║  Population     : %-5d                                     ║\n", cfg.population_size);
    printf("║  Generations    : %-5d                                     ║\n", cfg.generations);
    printf("║  m range        : [%d, %d]                                 ║\n", cfg.m_min, cfg.m_max);
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    // ── Initialise population ─────────────────────────────────────────────────
    std::vector<Individual> pop(cfg.population_size);
    for (int i = 0; i < cfg.population_size; i++) {
        pop[i] = random_individual(cfg, seed + (unsigned)i * 31337);
    }

    GAResult result;
    result.fitness_history.reserve(cfg.generations);

    // ── Generational loop ─────────────────────────────────────────────────────
    for (int gen = 0; gen < cfg.generations; gen++) {
        float temperature = 1.0f - (float)gen / (float)cfg.generations;

        // ── Evaluate population (sequential GPU calls) ────────────────────────
        std::vector<FitnessScore> scores(cfg.population_size);
        for (int i = 0; i < cfg.population_size; i++) {
            scores[i] = evaluate_individual(pop[i], h_T, n);
        }

        // ── Sort by fitness (descending) ──────────────────────────────────────
        std::vector<int> idx(cfg.population_size);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](int a, int b) {
            return pop[a].fitness > pop[b].fitness;
        });

        float best_fitness = pop[idx[0]].fitness;
        result.fitness_history.push_back(best_fitness);

        if (cfg.verbose) {
            printf("Gen %3d │ best=%.4f │ temp=%.3f │ ", gen, best_fitness, temperature);
            print_fitness(pop[idx[0]], scores[idx[0]]);
        }

        // ── Build next generation ─────────────────────────────────────────────
        std::vector<Individual> next_pop;
        next_pop.reserve(cfg.population_size);

        // Elitism: carry top-N unchanged
        for (int e = 0; e < cfg.elite_count && e < cfg.population_size; e++) {
            next_pop.push_back(pop[idx[e]]);
        }

        // Fill remainder with tournament → crossover → mutation
        while ((int)next_pop.size() < cfg.population_size) {
            Individual pa = tournament_select(pop, cfg.tournament_k);
            Individual pb = tournament_select(pop, cfg.tournament_k);
            Individual child = crossover(pa, pb, cfg);
            child = mutate(child, temperature, cfg);
            next_pop.push_back(child);
        }

        pop = std::move(next_pop);
    }

    // ── Final evaluation of the best individual ───────────────────────────────
    // Re-run STOMP one last time with verbose fitness breakdown
    Individual& best = pop[0];
    // Find actual best after last generation
    for (auto& ind : pop) {
        if (ind.fitness < 0.0f) {
            evaluate_individual(ind, h_T, n);
        }
    }
    std::sort(pop.begin(), pop.end(), [](const Individual& a, const Individual& b) {
        return a.fitness > b.fitness;
    });

    result.best_individual = pop[0];

    STOMPConfig best_cfg;
    best_cfg.window_size     = pop[0].window_size;
    best_cfg.ez_factor       = pop[0].ez_factor;
    best_cfg.min_motif_count = pop[0].min_motif_count;
    STOMPResult best_res = run_stomp(h_T, n, best_cfg);
    result.best_fitness = evaluate_fitness(best_res.mp, best_res.profile_len, pop[0]);
    free(best_res.mp);
    free(best_res.mpi);

    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║                    OPTIMISATION RESULT                      ║\n");
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║  Best window_size       : %-5d                             ║\n",
           result.best_individual.window_size);
    printf("║  Best ez_factor         : %.3f                             ║\n",
           result.best_individual.ez_factor);
    printf("║  Best min_motif_count   : %-5d                             ║\n",
           result.best_individual.min_motif_count);
    printf("║  Composite fitness      : %.4f                            ║\n",
           result.best_fitness.composite);
    printf("║    ↳ Contrast           : %.4f                            ║\n",
           result.best_fitness.contrast);
    printf("║    ↳ Entropy            : %.4f                            ║\n",
           result.best_fitness.entropy);
    printf("║    ↳ Count score        : %.4f                            ║\n",
           result.best_fitness.count_score);
    printf("║    ↳ Discovered motifs  : %-5d                             ║\n",
           result.best_fitness.discovered_motifs);
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    return result;
}
