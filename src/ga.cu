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
// -----------------------------------------------------------------------------
//  Random individual (with optional period bias)
// -----------------------------------------------------------------------------
Individual random_individual(const GAConfig &cfg, unsigned int seed, int detected_period = -1)
{
    srand(seed);
    Individual ind;
    
    if (detected_period > 0 && detected_period >= cfg.m_min && detected_period <= cfg.m_max)
    {
        // Bias window_size toward detected period using Gaussian distribution
        // Sigma = 20% of range, clamped to bounds
        float sigma = 0.20f * (float)(cfg.m_max - cfg.m_min);
        float gaussian_val = gaussianRand() * sigma + (float)detected_period;
        ind.window_size = clampVal((int)roundf(gaussian_val), cfg.m_min, cfg.m_max);
    }
    else
    {
        ind.window_size = cfg.m_min + rand() % (cfg.m_max - cfg.m_min + 1);
    }
    
    ind.ez_factor = cfg.ez_min +
                    (cfg.ez_max - cfg.ez_min) * ((float)rand() / (float)RAND_MAX);
    ind.min_motif_count = cfg.k_min + rand() % (cfg.k_max - cfg.k_min + 1);
    ind.fitness = -1.0f;
    return ind;
}

// -----------------------------------------------------------------------------
//  Tournament selection
// -----------------------------------------------------------------------------
Individual tournament_select(const std::vector<Individual> &pop, int k)
{
    Individual best = pop[rand() % pop.size()];
    for (int i = 1; i < k; i++)
    {
        Individual c = pop[rand() % pop.size()];
        if (c.fitness > best.fitness)
            best = c;
    }
    return best;
}

// -----------------------------------------------------------------------------
//  Arithmetic crossover
// -----------------------------------------------------------------------------
Individual crossover(const Individual &a, const Individual &b, const GAConfig &cfg)
{
    float alpha = 0.3f + 0.4f * ((float)rand() / (float)RAND_MAX);
    Individual child;
    child.window_size = clampVal(
        (int)roundf(alpha * (float)a.window_size + (1 - alpha) * (float)b.window_size),
        cfg.m_min, cfg.m_max);
    child.ez_factor = clampVal(
        alpha * a.ez_factor + (1 - alpha) * b.ez_factor,
        cfg.ez_min, cfg.ez_max);
    child.min_motif_count = (rand() % 2 == 0) ? a.min_motif_count : b.min_motif_count;
    child.fitness = -1.0f;
    return child;
}

// -----------------------------------------------------------------------------
//  Annealed Gaussian mutation
// -----------------------------------------------------------------------------
Individual mutate(Individual ind, float temperature, const GAConfig &cfg)
{
    if ((float)rand() / RAND_MAX < cfg.mutation_rate)
    {
        float sigma = (float)(cfg.m_max - cfg.m_min) * 0.10f * temperature;
        ind.window_size = clampVal(ind.window_size + (int)roundf(gaussianRand() * sigma),
                                   cfg.m_min, cfg.m_max);
    }
    if ((float)rand() / RAND_MAX < cfg.mutation_rate)
    {
        float sigma = (cfg.ez_max - cfg.ez_min) * 0.10f * temperature;
        ind.ez_factor = clampVal(ind.ez_factor + gaussianRand() * sigma,
                                 cfg.ez_min, cfg.ez_max);
    }
    if ((float)rand() / RAND_MAX < cfg.mutation_rate * 0.5f)
    {
        int delta = (rand() % 3) - 1;
        ind.min_motif_count = clampVal(ind.min_motif_count + delta, cfg.k_min, cfg.k_max);
    }
    ind.fitness = -1.0f;
    return ind;
}

// -----------------------------------------------------------------------------
//  Evaluate one individual
// -----------------------------------------------------------------------------
static FitnessScore evaluate_individual(
    Individual &ind,
    const float *h_T,
    int n,
    float approx_frac,
    int detected_period,  // FFT-detected period for reward signal
    TimingReport &timing)
{
    STOMPConfig scfg;
    scfg.window_size = ind.window_size;
    scfg.ez_factor = ind.ez_factor;
    scfg.min_motif_count = ind.min_motif_count;
    scfg.approximate = (approx_frac < 0.999f);
    scfg.approx_frac = approx_frac;
    scfg.recover_indices = false; // skip during GA -- not needed for fitness

    STOMPResult res = run_stomp(h_T, n, scfg);

    timing.stomp_total_ms += res.kernel_ms;
    timing.precompute_total_ms += res.precompute_ms;
    timing.n_stomp_calls++;

    FitnessScore fs{};
    if (res.mp)
    {
        WallTimer ft;
        ft.start();
        fs = evaluate_fitness(res.mp, res.profile_len, ind, detected_period);
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
    const float *h_T,
    int n,
    const GAConfig &cfg,
    unsigned int seed)
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

    // =========================================================================
    //  FFT Reference Period Detection (run once before GA)
    //  Uses multiple reference window sizes for robust period estimation
    // =========================================================================
    int detected_period = -1;
    {
        // Try multiple reference window sizes to get consensus period
        const int ref_windows[] = {30, 40, 50};
        const int n_refs = 3;
        std::vector<int> detected_periods;
        
        for (int wi = 0; wi < n_refs; wi++)
        {
            int w_ref = ref_windows[wi];
            printf("[fft] Computing reference MP at w_ref=%d for period detection...\n", w_ref);
            STOMPConfig ref_cfg;
            ref_cfg.window_size = w_ref;
            ref_cfg.ez_factor = 0.25f;
            ref_cfg.min_motif_count = 2;
            ref_cfg.approximate = true;
            ref_cfg.approx_frac = 0.5f;
            ref_cfg.recover_indices = false;

            STOMPResult ref_res = run_stomp(h_T, n, ref_cfg);
            if (ref_res.mp)
            {
                int period = detect_dominant_period(ref_res.mp, ref_res.profile_len, 30, 400);
                if (period > 0)
                {
                    detected_periods.push_back(period);
                    printf("[fft]   w_ref=%d -> detected period: %d samples\n", w_ref, period);
                }
                free(ref_res.mp);
                free(ref_res.mpi);
            }
            else
            {
                printf("[fft]   w_ref=%d -> Reference MP computation failed\n", w_ref);
            }
        }
        
        // Take the most common period (mode) or median if no clear mode
        if (!detected_periods.empty())
        {
            std::sort(detected_periods.begin(), detected_periods.end());
            // Simple mode detection: find most frequent value
            int mode = detected_periods[0];
            int mode_count = 1;
            int current = detected_periods[0];
            int current_count = 1;
            
            for (size_t i = 1; i < detected_periods.size(); i++)
            {
                if (detected_periods[i] == current)
                {
                    current_count++;
                }
                else
                {
                    if (current_count > mode_count)
                    {
                        mode = current;
                        mode_count = current_count;
                    }
                    current = detected_periods[i];
                    current_count = 1;
                }
            }
            if (current_count > mode_count)
            {
                mode = current;
                mode_count = current_count;
            }
            
            // If mode appears at least twice, use it; otherwise use median
            if (mode_count >= 2)
            {
                detected_period = mode;
            }
            else
            {
                detected_period = detected_periods[detected_periods.size() / 2];
            }
            
            printf("[fft] Consensus dominant period: %d samples (from %zu refs)\n", 
                   detected_period, detected_periods.size());
        }
        else
        {
            printf("[fft] All reference MP computations failed\n");
        }
    }

    // Initialise population
    std::vector<Individual> pop(cfg.population_size);
    for (int i = 0; i < cfg.population_size; i++)
        pop[i] = random_individual(cfg, seed + (unsigned)i * 31337, detected_period);

    GAResult result;
    result.fitness_history.reserve(cfg.generations);
    TimingReport &timing = result.timing;
    timing = {};

    WallTimer total_wall;
    total_wall.start();

    for (int gen = 0; gen < cfg.generations; gen++)
    {
        float temperature = 1.0f - (float)gen / (float)cfg.generations;

        // Evaluate population
        std::vector<FitnessScore> scores(cfg.population_size);
        for (int i = 0; i < cfg.population_size; i++)
            scores[i] = evaluate_individual(pop[i], h_T, n, cfg.approx_frac, detected_period, timing);

        // Sort descending
        std::vector<int> idx(cfg.population_size);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](int a, int b)
                  { return pop[a].fitness > pop[b].fitness; });

        float best_fitness = pop[idx[0]].fitness;
        result.fitness_history.push_back(best_fitness);

        if (cfg.verbose)
        {
            printf("Gen %3d | best=%.4f | temp=%.3f | ", gen, best_fitness, temperature);
            print_fitness(pop[idx[0]], scores[idx[0]]);
        }

        // Build next generation
        WallTimer ga_timer;
        ga_timer.start();
        std::vector<Individual> next_pop;
        next_pop.reserve(cfg.population_size);

        for (int e = 0; e < cfg.elite_count && e < cfg.population_size; e++)
            next_pop.push_back(pop[idx[e]]);

        while ((int)next_pop.size() < cfg.population_size)
        {
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
    for (auto &ind : pop)
        if (ind.fitness < 0.0f)
            evaluate_individual(ind, h_T, n, 1.0f, detected_period, timing);

    std::sort(pop.begin(), pop.end(), [](const Individual &a, const Individual &b)
              { return a.fitness > b.fitness; });

    result.best_individual = pop[0];

    // Final best run: full mode + recover indices
    STOMPConfig best_cfg;
    best_cfg.window_size = pop[0].window_size;
    best_cfg.ez_factor = pop[0].ez_factor;
    best_cfg.min_motif_count = pop[0].min_motif_count;
    best_cfg.approximate = false;
    best_cfg.approx_frac = 1.0f;
    best_cfg.recover_indices = true; // fill MPI[] properly on final run

    STOMPResult best_res = run_stomp(h_T, n, best_cfg);
    result.best_fitness = evaluate_fitness(best_res.mp, best_res.profile_len, pop[0], detected_period);
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
    printf("|    -> Spacing Reg     : %.4f                              |\n",
           result.best_fitness.spacing_regularity);
    printf("|    -> Period Reward   : %.4f                              |\n",
           result.best_fitness.period_reward);
    printf("|    -> Spacing Consist : %.4f                              |\n",
           result.best_fitness.spacing_consistency);
    printf("|    -> Count score     : %.4f                              |\n",
           result.best_fitness.count_score);
    printf("|    -> Motifs found    : %-5d                               |\n",
           result.best_fitness.discovered_motifs);
    printf("+--------------------------------------------------------------+\n\n");
    timing.print();

    return result;
}
