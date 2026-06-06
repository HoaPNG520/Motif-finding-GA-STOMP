// =============================================================================
//  fitness.cu -- Unsupervised composite fitness for GA individuals
//
//  All computation is CPU-side; it receives the host-side MP array from
//  run_stomp and derives three independent quality signals.
//
//  Design rationale:
//    Without labels, we cannot use classification accuracy.  Instead we
//    proxy "motif quality" with three signals that are theoretically
//    motivated and empirically correlated with ground-truth motif recall
//    in synthetic benchmarks (see README SValidation).
// =============================================================================

#include "fitness.cuh"
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <vector>
#include <numeric>

// -----------------------------------------------------------------------------
//  Signal 1 -- Motif Contrast
// -----------------------------------------------------------------------------
//  contrast = (d_second - d_top) / mean(MP)
//
//  d_top    = MP minimum (best motif pair distance)
//  d_second = MP value at position k (the min_motif_count-th smallest value)
//             -- a proxy for the "background" distance level
//
//  High contrast -> top motif is far below background -> genuinely distinctive.
//  Low contrast  -> many near-trivial matches -> m likely too small.
static float compute_contrast(
    const std::vector<float> &sorted_mp,
    int profile_len,
    int min_motif_count)
{
    float d_top = sorted_mp[0];
    int second_k = std::min(min_motif_count, profile_len - 1);
    float d_second = sorted_mp[second_k];

    // Mean MP value
    float mean_mp = 0.0f;
    for (float v : sorted_mp)
        mean_mp += v;
    mean_mp /= (float)sorted_mp.size();

    float contrast = (d_second - d_top) / (mean_mp + EPS);

    // Normalise to [0, 1] with a soft saturation at contrast = 2.0
    return 1.0f - expf(-contrast);
}

// -----------------------------------------------------------------------------
//  Signal 2 -- Motif Count Validity
// -----------------------------------------------------------------------------
//  Count subsequences whose MP distance is within a relative threshold of
//  the global minimum, then penalise the distance from the target k.
//
//  threshold = mp_min + 0.1 * (mean_mp - mp_min)
//  score = 1 - |discovered - k| / (k + EPS)
//
//  Rewards individuals where the threshold selects approximately k motifs.
static float compute_count_score(
    const float *mp,
    int profile_len,
    float mp_min, float mean_mp,
    int min_motif_count,
    int &discovered_out)
{
    float threshold = mp_min + 0.10f * (mean_mp - mp_min + EPS);
    int discovered = 0;
    for (int i = 0; i < profile_len; i++)
    {
        if (mp[i] < threshold)
            discovered++;
    }
    discovered_out = discovered;

    float diff = fabsf((float)discovered - (float)min_motif_count);
    float score = 1.0f - diff / ((float)min_motif_count + EPS);
    return fmaxf(0.0f, score);
}
// -----------------------------------------------------------------------------
//  Signal 4 -- Autocorrelation Peak Alignment (Approach 3)
// -----------------------------------------------------------------------------
// Computes the autocorrelation of the MP, finds its highest peak (the intrinsic
// defect period), and rewards window sizes that align with that peak.
static float compute_autocorr_peak(
    const float *mp,
    int profile_len,
    int w)
{
    // Mean-center the profile
    float mean = 0.0f;
    for (int i = 0; i < profile_len; i++)
        mean += mp[i];
    mean /= (float)profile_len;

    // Cap the search to avoid tail noise
    int max_lag = profile_len / 4;
    std::vector<float> ac(max_lag, 0.0f);
    float denom = 0.0f;

    for (int i = 0; i < profile_len; i++)
    {
        float c = mp[i] - mean;
        denom += c * c;
        for (int lag = 1; lag < max_lag && (i + lag) < profile_len; lag++)
        {
            ac[lag] += c * (mp[i + lag] - mean);
        }
    }

    // Normalize and find the peak
    int search_lo = 8; // Ignore trivial self-matches
    int peak_lag = search_lo;
    for (int lag = search_lo; lag < max_lag; lag++)
    {
        ac[lag] /= (denom + EPS);
        if (ac[lag] > ac[peak_lag])
            peak_lag = lag;
    }

    // Reward w if it aligns with the discovered peak_lag
    float diff = (float)(w - peak_lag);
    float sigma = (float)peak_lag * 0.30f;
    return expf(-0.5f * (diff / sigma) * (diff / sigma));
}

// -----------------------------------------------------------------------------
//  Main fitness evaluator
// -----------------------------------------------------------------------------
FitnessScore evaluate_fitness(
    const float *mp,
    int profile_len,
    const Individual &ind)
{
    FitnessScore fs{};

    if (profile_len <= 0 || mp == nullptr)
    {
        fs.composite = -1.0f;
        return fs;
    }

    // Sort a copy of MP for rank-based statistics
    std::vector<float> sorted_mp(mp, mp + profile_len);
    std::sort(sorted_mp.begin(), sorted_mp.end());

    float mp_min = sorted_mp.front();
    float mp_max = sorted_mp.back();

    // Mean (needed by multiple signals)
    float mean_mp = 0.0f;
    for (int i = 0; i < profile_len; i++)
        mean_mp += mp[i];
    mean_mp /= (float)profile_len;

    // Degenerate guard: if all values are INF or identical, skip
    if (mp_max < EPS || (mp_max - mp_min) < EPS)
    {
        fs.composite = 0.0f;
        return fs;
    }

    // -- Three signals ---------------------------------------------------------
    fs.contrast = compute_contrast(sorted_mp, profile_len,
                                   ind.min_motif_count);
    fs.count_score = compute_count_score(mp, profile_len,
                                         mp_min, mean_mp,
                                         ind.min_motif_count,
                                         fs.discovered_motifs);
    fs.autocorr_peak = compute_autocorr_peak(mp, profile_len, ind.window_size);

    // -- Composite weighted sum (v3 Formulation)------------------------------------------------
    fs.composite = 0.40f * fs.contrast + 0.40f * fs.autocorr_peak + 0.20f * fs.count_score;

    return fs;
}

// -----------------------------------------------------------------------------
//  Logging helper
// -----------------------------------------------------------------------------
void print_fitness(const Individual &ind, const FitnessScore &fs)
{
    printf("  Individual: m=%-4d  ez=%.2f  k=%-3d  "
           "| fit=%.4f  (contrast=%.3f  autocorr_peak=%.3f  count=%.3f  found=%d)\n",
           ind.window_size, ind.ez_factor, ind.min_motif_count,
           fs.composite, fs.contrast, fs.autocorr_peak, fs.count_score,
           fs.discovered_motifs);
}
