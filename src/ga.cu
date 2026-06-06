// =============================================================================
//  ga.cu -- Genetic Algorithm with timing instrumentation
// =============================================================================

#include "ga.cuh"
#include "stomp.cuh"
#include "fitness.cuh"
#include "timer.cuh"
#include "common.cuh"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>
#include <numeric>

// -----------------------------------------------------------------------------
//  Random individual
// -----------------------------------------------------------------------------
Individual random_individual(const GAConfig& cfg, unsigned int seed)
{
    srand(seed);
    Individual ind;
    ind.window_size     = cfg.m_min + rand() % (cfg.m_max - cfg.m_min + 1);
    ind.ez_factor       = cfg.ez_min +
        (cfg.ez_max - cfg.ez_min) * ((float)rand() / (float)RAND_MAX);
    ind.min_motif_count = cfg.k_min + rand() % (cfg.k_max - cfg.k_min + 1);
    ind.fitness         = -1.0f;
    return ind;
}

// -----------------------------------------------------------------------------
//  Tournament selection
// -----------------------------------------------------------------------------
Individual tournament_select(const std::vector<Individual>& pop, int k)
{
    Individual best = pop[rand() % pop.size()];
    for (int i = 1; i < k; i++) {
        Individual c = pop[rand() % pop.size()];
        if (c.fitness > best.fitness) best = c;
    }
    return best;
}

// -----------------------------------------------------------------------------
//  Arithmetic crossover
// -----------------------------------------------------------------------------
Individual crossover(const Individual& a, const Individual& b, const GAConfig& cfg)
{
    float alpha = 0.3f + 0.4f * ((float)rand() / (float)RAND_MAX);
    Individual child;
    child.window_size = clampVal(
        (int)roundf(alpha*(float)a.window_size + (1-alpha)*(float)b.window_size),
        cfg.m_min, cfg.m_max);
    child.ez_factor = clampVal(
        alpha*a.ez_factor + (1-alpha)*b.ez_factor,
        cfg.ez_min, cfg.ez_max);
    child.min_motif_count = (rand()%2==0) ? a.min_motif_count : b.min_motif_count;
    child.fitness = -1.0f;
    return child;
}

// -----------------------------------------------------------------------------
//  Annealed Gaussian mutation
// -----------------------------------------------------------------------------
Individual mutate(Individual ind, float temperature, const GAConfig& cfg)
{
    if ((float)rand()/RAND_MAX < cfg.mutation_rate) {
        float sigma = (float)(cfg.m_max - cfg.m_min) * 0.10f * temperature;
        ind.window_size = clampVal(ind.window_size + (int)roundf(gaussianRand()*sigma),
                                    cfg.m_min, cfg.m_max);
    }
    if ((float)rand()/RAND_MAX < cfg.mutation_rate) {
        float sigma = (cfg.ez_max - cfg.ez_min) * 0.10f * temperature;
        ind.ez_factor = clampVal(ind.ez_factor + gaussianRand()*sigma,
                                  cfg.ez_min, cfg.ez_max);
    }
    if ((float)rand()/RAND_MAX < cfg.mutation_rate*0.5f) {
        int delta = (rand()%3) - 1;
        ind.min_motif_count = clampVal(ind.min_motif_count + delta, cfg.k_min, cfg.k_max);
    }
    ind.fitness = -1.0f;
    return ind;
}

// -----------------------------------------------------------------------------
//  Evaluate one individual
// -----------------------------------------------------------------------------
static FitnessScore evaluate_individual(
    Individual&  ind,
    const float* h_T,
    int          n,
    float        approx_frac,
    TimingReport& timing)
{
    STOMPConfig scfg;
    scfg.window_size     = ind.window_size;
    scfg.ez_factor       = ind.ez_factor;
    scfg.min_motif_count = ind.min_motif_count;
    scfg.approximate     = (approx_frac < 0.999f);
    scfg.approx_frac     = approx_frac;
    scfg.recover_indices = false;   // skip during GA -- not needed for fitness

    STOMPResult res = run_stomp(h_T, n, scfg);

    timing.stomp_total_ms     += res.kernel_ms;
    timing.precompute_total_ms += res.precompute_ms;
    timing.n_stomp_calls++;

    FitnessScore fs{};
    if (res.mp) {
        WallTimer ft; ft.start();
        fs = evaluate_fitness(res.mp, res.profile_len, ind);
        timing.fitness_total_ms += ft.stop();
        free(res.mp);
        free(res.mpi);
    }
    ind.fitness = fs.composite;
    return fs;
}

// -----------------------------------------------------------------------------
//  Main GA loop
// -----------------------------------------------------------------------------
GAResult run_ga(
    const float*    h_T,
    int             n,
    const GAConfig& cfg,
    unsigned int    seed)
{
    srand(seed);

    printf("\n+--------------------------------------------------------------+\n");
    printf("|        GA + STOMP  Hyperparameter Optimisation              |\n");
    printf("+--------------------------------------------------------------+\n");
    printf("|  Series length  : %-5d  | Seed: %-10u               |\n", n, seed);
    printf("|  Population     : %-5d  | Generations: %-5d              |\n",
           cfg.population_size, cfg.generations);
    printf("|  m range        : [%d, %d]  | approx_frac: %.2f           |\n",
           cfg.m_min, cfg.m_max, cfg.approx_frac);
    printf("+--------------------------------------------------------------+\n\n");

    // Initialise population
    std::vector<Individual> pop(cfg.population_size);
    for (int i = 0; i < cfg.population_size; i++)
        pop[i] = random_individual(cfg, seed + (unsigned)i * 31337);

    GAResult result;
    result.fitness_history.reserve(cfg.generations);
    TimingReport& timing = result.timing;
    timing = {};

    WallTimer total_wall; total_wall.start();

    for (int gen = 0; gen < cfg.generations; gen++) {
        float temperature = 1.0f - (float)gen / (float)cfg.generations;

        // Evaluate population
        std::vector<FitnessScore> scores(cfg.population_size);
        for (int i = 0; i < cfg.population_size; i++)
            scores[i] = evaluate_individual(pop[i], h_T, n, cfg.approx_frac, timing);

        // Sort descending
        std::vector<int> idx(cfg.population_size);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](int a, int b){
            return pop[a].fitness > pop[b].fitness;
        });

        float best_fitness = pop[idx[0]].fitness;
        result.fitness_history.push_back(best_fitness);

        if (cfg.verbose) {
            printf("Gen %3d | best=%.4f | temp=%.3f | ", gen, best_fitness, temperature);
            print_fitness(pop[idx[0]], scores[idx[0]]);
        }

        // Build next generation
        WallTimer ga_timer; ga_timer.start();
        std::vector<Individual> next_pop;
        next_pop.reserve(cfg.population_size);

        for (int e = 0; e < cfg.elite_count && e < cfg.population_size; e++)
            next_pop.push_back(pop[idx[e]]);

        while ((int)next_pop.size() < cfg.population_size) {
            Individual pa = tournament_select(pop, cfg.tournament_k);
            Individual pb = tournament_select(pop, cfg.tournament_k);
            Individual child = crossover(pa, pb, cfg);
            child = mutate(child, temperature, cfg);
            next_pop.push_back(child);
        }
        timing.ga_overhead_ms += ga_timer.stop();

        pop = std::move(next_pop);
    }

    timing.total_wall_ms = total_wall.stop();

    // Final evaluation with full STOMP + MPI recovery
    for (auto& ind : pop)
        if (ind.fitness < 0.0f)
            evaluate_individual(ind, h_T, n, 1.0f, timing);

    std::sort(pop.begin(), pop.end(), [](const Individual& a, const Individual& b){
        return a.fitness > b.fitness;
    });

    result.best_individual = pop[0];

    // Final best run: full mode + recover indices
    STOMPConfig best_cfg;
    best_cfg.window_size     = pop[0].window_size;
    best_cfg.ez_factor       = pop[0].ez_factor;
    best_cfg.min_motif_count = pop[0].min_motif_count;
    best_cfg.approximate     = false;
    best_cfg.approx_frac     = 1.0f;
    best_cfg.recover_indices = true;   // fill MPI[] properly on final run

    STOMPResult best_res = run_stomp(h_T, n, best_cfg);
    result.best_fitness = evaluate_fitness(best_res.mp, best_res.profile_len, pop[0]);
    free(best_res.mp);
    free(best_res.mpi);

    printf("\n+--------------------------------------------------------------+\n");
    printf("|                   OPTIMISATION RESULT                       |\n");
    printf("+--------------------------------------------------------------+\n");
    printf("|  Best window_size     : %-5d                               |\n",
           result.best_individual.window_size);
    printf("|  Best ez_factor       : %.3f                               |\n",
           result.best_individual.ez_factor);
    printf("|  Best min_motif_count : %-5d                               |\n",
           result.best_individual.min_motif_count);
    printf("|  Composite fitness    : %.4f                              |\n",
           result.best_fitness.composite);
    printf("|    -> Contrast        : %.4f                              |\n",
           result.best_fitness.contrast);
    printf("|    -> Autocorr Peak   : %.4f                              |\n",
           result.best_fitness.autocorr_peak);
    printf("|    -> Count score     : %.4f                              |\n",
           result.best_fitness.count_score);
    printf("|    -> Motifs found    : %-5d                               |\n",
           result.best_fitness.discovered_motifs);
    printf("+--------------------------------------------------------------+\n\n");

    timing.print();

    return result;
}
