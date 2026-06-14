// =============================================================================
//  fitness.cu  --  Unsupervised composite fitness for GA-STOMP individuals
//
//  All computation is CPU-side; it receives the host-side MP array from
//  run_stomp and derives two independent quality signals.
//
//  Composite:
//    F = 0.35 × contrast + 0.65 × regularity
//
//  See fitness.cuh for full design rationale.
// =============================================================================

#include "fitness.cuh"
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <vector>
#include <numeric>

// =============================================================================
//  Signal 1 -- Motif Contrast
// =============================================================================
//  Measures how far the best motif stands out from the background:
//
//    contrast = (d_second − d_top) / mean(MP)
//    score    = 1 − exp(−contrast)          ← soft clamp to [0, 1]
//
//  d_top    = MP minimum  (best motif pair distance)
//  d_second = k-th smallest MP value  (proxy for background distance level)
//
//  High contrast → top motif is far below background → genuinely distinctive.
//  Low contrast  → many near-trivial matches → window likely too small.
// =============================================================================
static float compute_contrast(
    const std::vector<float> &sorted_mp,
    int   profile_len,
    int   min_motif_count)
{
    float d_top    = sorted_mp[0];
    int   second_k = std::min(min_motif_count, profile_len - 1);
    float d_second = sorted_mp[second_k];

    float mean_mp = 0.0f;
    for (float v : sorted_mp)
        mean_mp += v;
    mean_mp /= (float)sorted_mp.size();

    float contrast = (d_second - d_top) / (mean_mp + EPS);
    return 1.0f - expf(-contrast);   // soft normalisation to [0, 1]
}

// =============================================================================
//  Signal 2 -- Spacing Regularity  (fixed-count selection)
// =============================================================================
//  Selects the FIXED_COUNT lowest-MP positions (most repetitive subsequences)
//  and measures the Coefficient of Variation (CV) of the gaps between them.
//
//  Key design choice: FIXED COUNT, not a percentage.
//    A percentage threshold (e.g. top 5%) selects profile_len/20 positions
//    regardless of window size, giving mean_gap ≈ 20 for every m.  This makes
//    the CV signal window-size-blind and causes systematic collapse to m_min.
//
//    Using FIXED_COUNT = 50 decouples mean_gap from m:
//      mean_gap ≈ profile_len / 50  ≈  118 samples  (for n=6000, any m)
//    The CV then purely reflects whether those 50 positions recur periodically.
//
//  Physical interpretation:
//    At the true defect window size, bearing fault impulses repeat at the
//    defect frequency → gaps cluster tightly → low CV → high regularity.
//    At wrong window sizes, the top-50 positions are scattered → high CV.
//
//  regularity = 1 / (1 + CV)   ∈ (0, 1]
// =============================================================================
static void compute_spacing_regularity(
    const float              *mp,
    const std::vector<float> &sorted_mp,
    int                       profile_len,
    float                    &regularity_out)
{
    // Minimum positions needed for a meaningful CV estimate
    const int FIXED_COUNT = 50;

    if (profile_len < FIXED_COUNT * 2)
    {
        regularity_out = 0.0f;
        return;
    }

    // Select the FIXED_COUNT lowest-MP positions
    float threshold = sorted_mp[std::min(FIXED_COUNT, profile_len - 1)];

    std::vector<int> motif_idx;
    motif_idx.reserve(FIXED_COUNT + 16);  // slight overalloc for tie-breaking
    for (int i = 0; i < profile_len; i++)
    {
        if (mp[i] <= threshold)
            motif_idx.push_back(i);
    }

    if ((int)motif_idx.size() < 4)
    {
        regularity_out = 0.0f;
        return;
    }

    // Compute gaps between consecutive selected positions
    std::vector<float> gaps;
    gaps.reserve(motif_idx.size() - 1);
    for (size_t i = 1; i < motif_idx.size(); i++)
        gaps.push_back((float)(motif_idx[i] - motif_idx[i - 1]));

    // Mean gap
    float mean_gap = 0.0f;
    for (float g : gaps)
        mean_gap += g;
    mean_gap /= (float)gaps.size();

    // Standard deviation of gaps
    float var = 0.0f;
    for (float g : gaps)
        var += (g - mean_gap) * (g - mean_gap);
    float std_gap = sqrtf(var / (float)gaps.size());

    // CV → regularity in [0, 1]
    float cv = std_gap / (mean_gap + EPS);
    regularity_out = 1.0f / (1.0f + cv);
}

// =============================================================================
//  Legacy helper -- Motif Count Score  (not used in composite)
// =============================================================================
//  Kept for logging and ablation studies.  Removed from composite because
//  it is gameable: the GA can always match discovered == k by tuning k,
//  earning count_score = 1.0 for any window size.
// =============================================================================
static float compute_count_score(
    const float *mp,
    int          profile_len,
    float        mp_min,
    float        mean_mp,
    int          min_motif_count,
    int         &discovered_out)
{
    float threshold  = mp_min + 0.10f * (mean_mp - mp_min + EPS);
    int   discovered = 0;
    for (int i = 0; i < profile_len; i++)
        if (mp[i] < threshold)
            discovered++;
    discovered_out = discovered;

    float diff  = fabsf((float)discovered - (float)min_motif_count);
    float score = 1.0f - diff / ((float)min_motif_count + EPS);
    return fmaxf(0.0f, score);
}

// =============================================================================
//  Main fitness evaluator
// =============================================================================
FitnessScore evaluate_fitness(
    const float      *mp,
    int               profile_len,
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

    // Guard: degenerate profile (all INF or all identical)
    if (mp_max < EPS || (mp_max - mp_min) < EPS)
    {
        fs.composite = 0.0f;
        return fs;
    }

    // Mean MP value (needed by contrast and count_score)
    float mean_mp = 0.0f;
    for (int i = 0; i < profile_len; i++)
        mean_mp += mp[i];
    mean_mp /= (float)profile_len;

    // --- Compute signals -------------------------------------------------------

    fs.contrast = compute_contrast(sorted_mp, profile_len, ind.min_motif_count);

    compute_spacing_regularity(mp, sorted_mp, profile_len, fs.spacing_regularity);

    // Legacy: count_score for logging / ablation only
    fs.count_score = compute_count_score(
        mp, profile_len, mp_min, mean_mp,
        ind.min_motif_count, fs.discovered_motifs);
    fs.spacing_consistency = fs.spacing_regularity;  // alias for display

    // --- Composite fitness -----------------------------------------------------
    //
    //   F = 0.35 × contrast + 0.65 × regularity
    //
    //   contrast   (0.35): ensures the GA does not ignore motif quality entirely.
    //   regularity (0.65): dominant signal; measures periodic recurrence of the
    //                      top-50 most-repetitive positions in the MP.
    //
    fs.composite = 0.35f * fs.contrast + 0.65f * fs.spacing_regularity;
    return fs;
}

// =============================================================================
//  Logging helper
// =============================================================================
void print_fitness(const Individual &ind, const FitnessScore &fs)
{
    printf("  Individual: m=%-4d  ez=%.2f  k=%-3d  "
           "| fit=%.4f  (contrast=%.3f  reg=%.3f  consist=%.3f)\n",
           ind.window_size, ind.ez_factor, ind.min_motif_count,
           fs.composite, fs.contrast, fs.spacing_regularity, fs.spacing_consistency);
}
