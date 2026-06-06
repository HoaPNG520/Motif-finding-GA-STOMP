// =============================================================================
//  fitness.cu -- Unsupervised composite fitness for GA individuals (v7 Combined)
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
static float compute_contrast(const std::vector<float> &sorted_mp, int profile_len, int min_motif_count)
{
    float d_top = sorted_mp[0];
    int second_k = std::min(min_motif_count, profile_len - 1);
    float d_second = sorted_mp[second_k];

    float mean_mp = 0.0f;
    for (float v : sorted_mp)
        mean_mp += v;
    mean_mp /= (float)sorted_mp.size();

    float contrast = (d_second - d_top) / (mean_mp + EPS);
    return 1.0f - expf(-contrast);
}

// -----------------------------------------------------------------------------
//  Signal 2 -- Motif Count Validity
// -----------------------------------------------------------------------------
static float compute_count_score(const float *mp, int profile_len, float mp_min, float mean_mp, int min_motif_count, int &discovered_out)
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
//  Signal 3 -- Autocorrelation Peak Alignment (From v3)
// -----------------------------------------------------------------------------
static float compute_autocorr_peak(const float *mp, int profile_len, int w)
{
    float mean = 0.0f;
    for (int i = 0; i < profile_len; i++)
        mean += mp[i];
    mean /= (float)profile_len;

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

    int search_lo = 8;
    int peak_lag = search_lo;
    for (int lag = search_lo; lag < max_lag; lag++)
    {
        ac[lag] /= (denom + EPS);
        if (ac[lag] > ac[peak_lag])
            peak_lag = lag;
    }

    float diff = (float)(w - peak_lag);
    float sigma = (float)peak_lag * 0.30f;
    return expf(-0.5f * (diff / sigma) * (diff / sigma));
}

// -----------------------------------------------------------------------------
//  Signal 4 & 5 -- Spacing Regularity & Consistency (From v6)
// -----------------------------------------------------------------------------
static void compute_spacing_metrics(const float *mp, const std::vector<float> &sorted_mp, int profile_len, int w, float &regularity_out, float &consistency_out)
{
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

    std::vector<float> gaps;
    for (size_t i = 1; i < motif_idx.size(); i++)
    {
        gaps.push_back((float)(motif_idx[i] - motif_idx[i - 1]));
    }

    float mean_gap = 0.0f, var = 0.0f;
    for (float g : gaps)
        mean_gap += g;
    mean_gap /= gaps.size();
    for (float g : gaps)
        var += (g - mean_gap) * (g - mean_gap);

    float std_gap = sqrtf(var / gaps.size());
    float cv = std_gap / (mean_gap + EPS);
    regularity_out = 1.0f / (1.0f + cv);

    float diff = mean_gap - (float)w;
    float sigma = (float)w * 0.50f;
    consistency_out = expf(-0.5f * (diff / sigma) * (diff / sigma));
}

// -----------------------------------------------------------------------------
//  Main fitness evaluator
// -----------------------------------------------------------------------------
FitnessScore evaluate_fitness(const float *mp, int profile_len, const Individual &ind)
{
    FitnessScore fs{};

    if (profile_len <= 0 || mp == nullptr)
    {
        fs.composite = -1.0f;
        return fs;
    }

    std::vector<float> sorted_mp(mp, mp + profile_len);
    std::sort(sorted_mp.begin(), sorted_mp.end());

    float mp_min = sorted_mp.front();
    float mp_max = sorted_mp.back();

    float mean_mp = 0.0f;
    for (int i = 0; i < profile_len; i++)
        mean_mp += mp[i];
    mean_mp /= (float)profile_len;

    if (mp_max < EPS || (mp_max - mp_min) < EPS)
    {
        fs.composite = 0.0f;
        return fs;
    }

    // -- Compute All Signals --
    fs.contrast = compute_contrast(sorted_mp, profile_len, ind.min_motif_count);
    fs.count_score = compute_count_score(mp, profile_len, mp_min, mean_mp, ind.min_motif_count, fs.discovered_motifs);
    fs.autocorr_peak = compute_autocorr_peak(mp, profile_len, ind.window_size);
    compute_spacing_metrics(mp, sorted_mp, profile_len, ind.window_size, fs.spacing_regularity, fs.spacing_consistency);

    // -- V7 Combined Formulation --
    fs.composite = 0.30f * fs.contrast + 0.25f * fs.autocorr_peak + 0.20f * fs.spacing_regularity + 0.15f * fs.spacing_consistency + 0.10f * fs.count_score;

    return fs;
}

// -----------------------------------------------------------------------------
//  Logging helper
// -----------------------------------------------------------------------------
void print_fitness(const Individual &ind, const FitnessScore &fs)
{
    printf("  Individual: m=%-4d ez=%.2f k=%-3d "
           "| fit=%.4f (cont=%.2f ac=%.2f reg=%.2f cons=%.2f count=%.2f)\n",
           ind.window_size, ind.ez_factor, ind.min_motif_count,
           fs.composite, fs.contrast, fs.autocorr_peak, fs.spacing_regularity, fs.spacing_consistency, fs.count_score);
}