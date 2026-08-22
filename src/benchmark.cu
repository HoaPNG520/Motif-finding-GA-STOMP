// =============================================================================
//  benchmark.cu -- Multi-seed variance report and fitness-recall sweep
// =============================================================================

#include "benchmark.cuh"
#include "stomp.cuh"
#include "fitness.cuh"
#include "timer.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#include <numeric>

// -----------------------------------------------------------------------------
//  MultiseedReport -- statistics
// -----------------------------------------------------------------------------
static float vec_mean(const std::vector<float> &v)
{
    float s = 0;
    for (float x : v)
        s += x;
    return s / (float)v.size();
}
static float vec_std(const std::vector<float> &v, float mean)
{
    float s = 0;
    for (float x : v)
        s += (x - mean) * (x - mean);
    return sqrtf(s / (float)v.size());
}

void MultiseedReport::finalize()
{
    std::vector<float> ms, ezs, fits, times;
    for (auto &r : results)
    {
        ms.push_back((float)r.best_m);
        ezs.push_back(r.best_ez);
        fits.push_back(r.best_fitness);
        times.push_back(r.wall_time_ms);
    }
    mean_m = vec_mean(ms);
    std_m = vec_std(ms, mean_m);
    mean_ez = vec_mean(ezs);
    std_ez = vec_std(ezs, mean_ez);
    mean_fit = vec_mean(fits);
    std_fit = vec_std(fits, mean_fit);
    mean_time = vec_mean(times);
    std_time = vec_std(times, mean_time);
}

void MultiseedReport::print() const
{
    printf("\n+--------------------------------------------------------------+\n");
    printf("|              MULTI-SEED REPRODUCIBILITY REPORT              |\n");
    printf("+--------------------------------------------------------------+\n");
    printf("| Seed  | best_m | best_ez | best_fit | wall_ms              |\n");
    printf("+-------+--------+---------+----------+----------------------+\n");
    for (auto &r : results)
    {
        printf("| %5u | %6d | %7.3f | %8.4f | %8.1f                |\n",
               r.seed, r.best_m, r.best_ez, r.best_fitness, r.wall_time_ms);
    }
    printf("+--------------------------------------------------------------+\n");
    printf("| mean  | %6.1f | %7.3f | %8.4f | %8.1f                |\n",
           mean_m, mean_ez, mean_fit, mean_time);
    printf("| std   | %6.1f | %7.3f | %8.4f | %8.1f                |\n",
           std_m, std_ez, std_fit, std_time);
    printf("+--------------------------------------------------------------+\n\n");
}

void MultiseedReport::save_csv(const char *path) const
{
    FILE *fp = fopen(path, "w");
    if (!fp)
    {
        fprintf(stderr, "[bench] Cannot write '%s'\n", path);
        return;
    }
    fprintf(fp, "seed,best_m,best_ez,best_k,best_fitness,wall_ms\n");
    for (auto &r : results)
    {
        fprintf(fp, "%u,%d,%.4f,%d,%.6f,%.2f\n",
                r.seed, r.best_m, r.best_ez, r.best_k,
                r.best_fitness, r.wall_time_ms);
    }
    fprintf(fp, "MEAN,%.2f,%.4f,,%.6f,%.2f\n",
            mean_m, mean_ez, mean_fit, mean_time);
    fprintf(fp, "STD,%.2f,%.4f,,%.6f,%.2f\n",
            std_m, std_ez, std_fit, std_time);
    fclose(fp);
    printf("[bench] Multi-seed report -> '%s'\n", path);
}

// -----------------------------------------------------------------------------
//  run_multiseed
// -----------------------------------------------------------------------------
MultiseedReport run_multiseed(
    const float *h_T,
    int n,
    const GAConfig &cfg,
    const unsigned int *seeds,
    int n_seeds)
{
    // Avoid circular dependency -- forward declare run_ga
    extern GAResult run_ga(const float *, int, const GAConfig &, unsigned int);

    MultiseedReport report;
    printf("\n[bench] Running multi-seed experiment: %d seeds\n\n", n_seeds);

    for (int s = 0; s < n_seeds; s++)
    {
        WallTimer wt;
        wt.start();
        GAResult res = run_ga(h_T, n, cfg, seeds[s]);
        float elapsed = wt.stop();

        SeedResult sr;
        sr.seed = seeds[s];
        sr.best_m = res.best_individual.window_size;
        sr.best_ez = res.best_individual.ez_factor;
        sr.best_k = res.best_individual.min_motif_count;
        sr.best_fitness = res.best_fitness.composite;
        sr.wall_time_ms = elapsed;
        report.results.push_back(sr);

        printf("[bench] Seed %u done: m=%d ez=%.3f fit=%.4f time=%.0fms\n",
               seeds[s], sr.best_m, sr.best_ez, sr.best_fitness, elapsed);
    }

    report.finalize();
    return report;
}

// -----------------------------------------------------------------------------
//  Motif recall helper
// -----------------------------------------------------------------------------
//  For a given Matrix Profile, find the index of its minimum.
//  Recall = 1.0 if that index falls within [true_pos, true_pos + true_m),
//           0.0 otherwise.  We check both injection sites.
static float compute_recall(
    const float *mp, int profile_len,
    int true_pos1, int true_pos2, int true_m)
{
    // Find top motif index
    int best = 0;
    for (int i = 1; i < profile_len; i++)
        if (mp[i] < mp[best])
            best = i;

    auto overlaps = [&](int idx, int pos) -> bool
    {
        return (idx >= pos - true_m && idx < pos + true_m);
    };

    if (overlaps(best, true_pos1) || overlaps(best, true_pos2))
        return 1.0f;
    return 0.0f;
}

// -----------------------------------------------------------------------------
//  Spearman rank correlation
// -----------------------------------------------------------------------------
static float spearman(const std::vector<float> &x, const std::vector<float> &y)
{
    int n = (int)x.size();
    if (n < 2)
        return 0.0f;

    // Rank x
    std::vector<int> rx(n), ry(n);
    std::vector<int> ix(n), iy(n);
    std::iota(ix.begin(), ix.end(), 0);
    std::iota(iy.begin(), iy.end(), 0);
    std::sort(ix.begin(), ix.end(), [&](int a, int b)
              { return x[a] < x[b]; });
    std::sort(iy.begin(), iy.end(), [&](int a, int b)
              { return y[a] < y[b]; });
    for (int i = 0; i < n; i++)
    {
        rx[ix[i]] = i;
        ry[iy[i]] = i;
    }

    float d2 = 0;
    for (int i = 0; i < n; i++)
    {
        float d = (float)(rx[i] - ry[i]);
        d2 += d * d;
    }
    return 1.0f - 6.0f * d2 / (float)(n * (n * n - 1));
}

// -----------------------------------------------------------------------------
//  run_fitness_recall_sweep
// -----------------------------------------------------------------------------
SweepReport run_fitness_recall_sweep(
    const float *h_T,
    int n,
    int true_m,
    int true_pos1,
    int true_pos2,
    int m_min,
    int m_max,
    int step_size,
    float ez_factor,
    int min_k)
{
    SweepReport report;

    printf("\n[bench] Fitness-recall sweep: m in [%d, %d] step %d\n\n",
           m_min, m_max, step_size);

    for (int m = m_min; m <= m_max; m += step_size)
    {
        STOMPConfig cfg;
        cfg.window_size = m;
        cfg.ez_factor = ez_factor;
        cfg.min_motif_count = min_k;
        cfg.approximate = false;
        cfg.recover_indices = false;

        STOMPResult res = run_stomp(h_T, n, cfg);
        if (!res.mp)
            continue;

        int profile_len = res.profile_len;

        // Build temporary Individual for fitness call
        Individual ind;
        ind.window_size = m;
        ind.ez_factor = ez_factor;
        ind.min_motif_count = min_k;
        ind.fitness = -1.0f;

        FitnessScore fs = evaluate_fitness(res.mp, profile_len, ind);

        float recall = compute_recall(res.mp, profile_len,
                                      true_pos1, true_pos2, true_m);

        SweepRow row;
        row.m = m;
        row.fitness = fs.composite;
        row.recall = recall;
        row.top_dist = *std::min_element(res.mp, res.mp + profile_len);
        row.regularity = fs.spacing_regularity;
        row.consistency = fs.spacing_consistency;
        row.contrast = fs.contrast;
        report.rows.push_back(row);

        printf("  m=%4d  fit=%.4f  recall=%.1f  top_dist=%.4f\n",
               m, fs.composite, recall, row.top_dist);

        free(res.mp);
        free(res.mpi);
    }

    // Compute Spearman rho between fitness and recall
    std::vector<float> fits, recalls;
    for (auto &r : report.rows)
    {
        fits.push_back(r.fitness);
        recalls.push_back(r.recall);
    }
    report.spearman_rho = spearman(fits, recalls);

    printf("\n[bench] Spearman rho (fitness vs recall) = %.4f\n", report.spearman_rho);
    return report;
}

void SweepReport::print() const
{
    printf("\n+------------------------------------------------------------------+\n");
    printf("|                  FITNESS-RECALL SWEEP RESULTS                    |\n");
    printf("+------+--------+--------+----------+----------+----------+--------+\n");
    printf("|  m   |  fit   | recall | top_dist |   reg    | consist  | contra |\n");
    printf("+------+--------+--------+----------+----------+----------+--------+\n");
    for (auto &r : rows)
    {
        printf("| %4d | %.4f |  %.1f   | %8.4f | %8.4f | %8.4f | %.4f |\n",
               r.m, r.fitness, r.recall, r.top_dist, r.regularity, r.consistency, r.contrast);
    }
    printf("+------------------------------------------------------------------+\n");
    printf("  Spearman rho (fitness vs recall): %.4f\n\n", spearman_rho);
}
void SweepReport::save_csv(const char *path) const
{
    FILE *fp = fopen(path, "w");
    if (!fp)
    {
        fprintf(stderr, "[bench] Cannot write '%s'\n", path);
        return;
    }
    fprintf(fp, "# Spearman rho = %.4f\n", spearman_rho);
    fprintf(fp, "m,fitness,recall,top_dist,regularity,consistency,contrast\n");
    for (auto &r : rows)
    {
        fprintf(fp, "%d,%.6f,%.1f,%.6f,%.6f,%.6f,%.6f\n",
                r.m, r.fitness, r.recall, r.top_dist, r.regularity, r.consistency, r.contrast);
    }
    fclose(fp);
    printf("[bench] Sweep report -> '%s'\n", path);
}
