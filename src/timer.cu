// =============================================================================
//  timer.cu -- TimingReport output implementations
// =============================================================================

#include "timer.cuh"
#include <cstdio>
#include <cerrno>
#include <cstring>

void TimingReport::print() const
{
    float stomp_pct  = (total_wall_ms > 0) ? 100.0f * stomp_total_ms   / total_wall_ms : 0;
    float pre_pct    = (total_wall_ms > 0) ? 100.0f * precompute_total_ms / total_wall_ms : 0;
    float fit_pct    = (total_wall_ms > 0) ? 100.0f * fitness_total_ms  / total_wall_ms : 0;
    float ga_pct     = (total_wall_ms > 0) ? 100.0f * ga_overhead_ms    / total_wall_ms : 0;
    float avg_stomp  = (n_stomp_calls > 0) ? stomp_total_ms / (float)n_stomp_calls : 0;

    printf("\n+--------------------------------------------------------------+\n");
    printf("|                    TIMING BREAKDOWN                         |\n");
    printf("+--------------------------------------------------------------+\n");
    printf("|  Total wall time        : %8.1f ms                       |\n", total_wall_ms);
    printf("|  STOMP kernel total     : %8.1f ms  (%4.1f%%)              |\n", stomp_total_ms, stomp_pct);
    printf("|  Pre-compute total      : %8.1f ms  (%4.1f%%)              |\n", precompute_total_ms, pre_pct);
    printf("|  Fitness eval total     : %8.1f ms  (%4.1f%%)              |\n", fitness_total_ms, fit_pct);
    printf("|  GA overhead total      : %8.1f ms  (%4.1f%%)              |\n", ga_overhead_ms, ga_pct);
    printf("|  STOMP calls            : %8d                            |\n", n_stomp_calls);
    printf("|  Avg ms / STOMP call    : %8.2f ms                       |\n", avg_stomp);
    printf("+--------------------------------------------------------------+\n\n");
}

void TimingReport::save_csv(const char* path) const
{
    FILE* fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr, "[timer] Cannot write '%s': %s\n", path, strerror(errno));
        return;
    }
    fprintf(fp, "metric,value_ms\n");
    fprintf(fp, "total_wall,%.3f\n",       total_wall_ms);
    fprintf(fp, "stomp_total,%.3f\n",      stomp_total_ms);
    fprintf(fp, "precompute_total,%.3f\n", precompute_total_ms);
    fprintf(fp, "fitness_total,%.3f\n",    fitness_total_ms);
    fprintf(fp, "ga_overhead,%.3f\n",      ga_overhead_ms);
    fprintf(fp, "n_stomp_calls,%d\n",      n_stomp_calls);
    fprintf(fp, "avg_per_stomp,%.3f\n",
            n_stomp_calls > 0 ? stomp_total_ms / (float)n_stomp_calls : 0.0f);
    fclose(fp);
    printf("[timer] Timing report -> '%s'\n", path);
}
