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
//  Signal 3 & 4 -- Spacing Regularity & Consistency (Approach 6)
// -----------------------------------------------------------------------------
// Finds the top 5% of motifs and measures the coefficient of variation (CV)
// of the gaps between them. Penalizes irregular spacing and penalizes gaps
// that do not match the proposed window size `w`.
static void compute_spacing_metrics(
    const float *mp,
    const std::vector<float> &sorted_mp,
    int profile_len,
    int w,
    float &regularity_out,
    float &consistency_out)
{
    // 5th percentile threshold
    float threshold = sorted_mp[profile_len / 20];

    std::vector<int> motif_idx;
    for (int i = 0; i < profile_len; i++)
    {
        if (mp[i] <= threshold)
            motif_idx.push_back(i);
    }

    if (motif_idx.size() < 4)
    {
        regularity_out = 0.0f;
        consistency_out = 0.0f;
        return;
    }

    // Compute gaps between consecutive motif occurrences
    std::vector<float> gaps;
    for (size_t i = 1; i < motif_idx.size(); i++)
    {
        gaps.push_back((float)(motif_idx[i] - motif_idx[i - 1]));
    }

    float mean_gap = 0.0f;
    for (float g : gaps)
        mean_gap += g;
    mean_gap /= gaps.size();

    float var = 0.0f;
    for (float g : gaps)
        var += (g - mean_gap) * (g - mean_gap);
    float std_gap = sqrtf(var / gaps.size());

    // Regularity: penalize high coefficient of variation
    float cv = std_gap / (mean_gap + EPS);
    regularity_out = 1.0f / (1.0f + cv);

    // Consistency: mean gap should approximate the window size
    float diff = mean_gap - (float)w;
    float sigma = (float)w * 0.50f;
    consistency_out = expf(-0.5f * (diff / sigma) * (diff / sigma));
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

    // -- Compute Signals ---------------------------------------------------------
    fs.contrast = compute_contrast(sorted_mp, profile_len, ind.min_motif_count);
    fs.count_score = compute_count_score(mp, profile_len, mp_min, mean_mp, ind.min_motif_count, fs.discovered_motifs);

    // NEW: Calculate Spacing Metrics
    compute_spacing_metrics(mp, sorted_mp, profile_len, ind.window_size, fs.spacing_regularity, fs.spacing_consistency);

    // -- Composite weighted sum (v6 Formulation) -------------------------------
    fs.composite = 0.40f * fs.contrast + 0.35f * fs.spacing_regularity + 0.25f * fs.spacing_consistency;
    return fs;
}

// -----------------------------------------------------------------------------
//  Logging helper
// -----------------------------------------------------------------------------
void print_fitness(const Individual &ind, const FitnessScore &fs)
{
    printf("  Individual: m=%-4d  ez=%.2f  k=%-3d  "
           "| fit=%.4f  (contrast=%.3f  reg=%.3f  consist=%.3f)\n",
           ind.window_size, ind.ez_factor, ind.min_motif_count,
           fs.composite, fs.contrast, fs.spacing_regularity, fs.spacing_consistency);
}
