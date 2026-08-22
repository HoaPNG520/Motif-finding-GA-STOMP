// =============================================================================
//  fitness.cu  --  Unsupervised composite fitness for GA-STOMP individuals
//
//  All computation is CPU-side; it receives the host-side MP array from
//  run_stomp and derives three independent quality signals.
//
//  Composite:
//    F = 0.25 × contrast + 0.40 × regularity + 0.35 × period_reward
//
//  See fitness.cuh for full design rationale.
// =============================================================================

#include "fitness.cuh"
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <vector>
#include <numeric>
#include <complex>
#include <valarray>

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
//  Signal 2 -- Spacing Regularity  (fixed-count selection, non-overlapping gaps)
// =============================================================================
//  Selects the FIXED_COUNT lowest-MP positions (most repetitive subsequences)
//  and measures the Coefficient of Variation (CV) of the NON-OVERLAPPING gaps
//  between them. A gap is only counted if it is >= window_size (w), ensuring
//  that adjacent selected positions belonging to the same motif instance do not
//  inflate regularity.
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
    int                       w,              // window size (for non-overlapping gap filter)
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

    // Compute NON-OVERLAPPING gaps between selected positions
    // Only count a gap if it is >= window_size (w), i.e. the two positions
    // belong to different motif instances, not the same one.
    std::vector<float> gaps;
    int last = motif_idx[0];
    for (size_t i = 1; i < motif_idx.size(); i++)
    {
        int gap = motif_idx[i] - last;
        if (gap >= w)
        {
            gaps.push_back((float)gap);
            last = motif_idx[i];
        }
    }

    // Require at least 4 non-overlapping gaps for a meaningful CV
    if ((int)gaps.size() < 4)
    {
        regularity_out = 0.0f;
        return;
    }

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
//  FFT-based Reference Period Detection
// =============================================================================
//  Computes the dominant periodicity in the Matrix Profile by:
//  1. Computing (mean_mp - mp) to get a "motif strength" signal
//  2. FFT of that signal to find the dominant frequency
//  3. Converting to period in samples
// =============================================================================

// Cooley-Tukey FFT (in-place, iterative)
static void fft(std::valarray<std::complex<float>> &x)
{
    const size_t N = x.size();
    if (N <= 1) return;

    // Bit-reversal permutation
    for (size_t i = 1, j = 0; i < N; i++)
    {
        size_t bit = N >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j)
            std::swap(x[i], x[j]);
    }

    // Iterative FFT
    for (size_t len = 2; len <= N; len <<= 1)
    {
        float ang = -2.0f * M_PI / (float)len;
        std::complex<float> wlen(cosf(ang), sinf(ang));
        for (size_t i = 0; i < N; i += len)
        {
            std::complex<float> w(1.0f, 0.0f);
            for (size_t j = 0; j < len / 2; j++)
            {
                std::complex<float> u = x[i + j];
                std::complex<float> v = x[i + j + len / 2] * w;
                x[i + j] = u + v;
                x[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
}

// Detect dominant period from MP using FFT
// Returns period in samples, or -1 if detection fails
int detect_dominant_period(const float *mp, int profile_len, int min_period, int max_period)
{
    if (profile_len < 64 || mp == nullptr)
        return -1;

    // Compute mean MP
    float mean_mp = 0.0f;
    for (int i = 0; i < profile_len; i++)
        mean_mp += mp[i];
    mean_mp /= (float)profile_len;

    // Create signal: mean_mp - mp (high values = strong motifs)
    // Pad to next power of 2 for FFT
    int N = 1;
    while (N < profile_len)
        N <<= 1;

    std::valarray<std::complex<float>> signal(N);
    for (int i = 0; i < profile_len; i++)
        signal[i] = std::complex<float>(mean_mp - mp[i], 0.0f);
    for (int i = profile_len; i < N; i++)
        signal[i] = std::complex<float>(0.0f, 0.0f);

    // FFT
    fft(signal);

    // Find peak in frequency domain (skip DC at index 0)
    // Frequency resolution: fs / N, where fs = 1 sample per index
    // Period = N / k for bin k
    float max_mag = 0.0f;
    int best_k = -1;

    for (int k = 1; k < N / 2; k++)
    {
        float period = (float)N / (float)k;
        if (period < min_period || period > max_period)
            continue;

        float mag = std::abs(signal[k]);
        if (mag > max_mag)
        {
            max_mag = mag;
            best_k = k;
        }
    }

    if (best_k <= 0)
        return -1;

    int period = N / best_k;
    return period;
}

// Gaussian reward centered on detected period
float compute_period_reward(int window_size, int detected_period, float sigma)
{
    if (detected_period <= 0)
        return 0.5f;  // neutral if no period detected

    float diff = (float)window_size - (float)detected_period;
    return expf(-0.5f * diff * diff / (sigma * sigma));
}

// =============================================================================
//  Main fitness evaluator
// =============================================================================
FitnessScore evaluate_fitness(
    const float      *mp,
    int               profile_len,
    const Individual &ind,
    int               detected_period)  // -1 = no period reward
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

    compute_spacing_regularity(mp, sorted_mp, profile_len, ind.window_size, fs.spacing_regularity);

    // Period reward (Signal 3)
    if (detected_period > 0)
    {
        // Sigma = 20% of detected period (allows ±40% for ~95% of reward)
        float sigma = 0.20f * (float)detected_period;
        fs.period_reward = compute_period_reward(ind.window_size, detected_period, sigma);
    }
    else
    {
        fs.period_reward = 0.5f;  // neutral
    }

    // Legacy: count_score for logging / ablation only
    fs.count_score = compute_count_score(
        mp, profile_len, mp_min, mean_mp,
        ind.min_motif_count, fs.discovered_motifs);
    fs.spacing_consistency = fs.spacing_regularity;  // alias for display

    // --- Composite fitness -----------------------------------------------------
    //
    //   F = 0.20 × contrast + 0.30 × regularity + 0.50 × period_reward
    //
    //   contrast       (0.20): ensures the GA does not ignore motif quality entirely.
    //   regularity     (0.30): measures periodic recurrence of the top-50 positions.
    //   period_reward  (0.50): strongly guides toward FFT-detected defect frequency.
    //
    fs.composite = 0.20f * fs.contrast + 0.30f * fs.spacing_regularity + 0.50f * fs.period_reward;
    return fs;
}

// =============================================================================
//  Logging helper
// =============================================================================
void print_fitness(const Individual &ind, const FitnessScore &fs)
{
    printf("  Individual: m=%-4d  ez=%.2f  k=%-3d  "
           "| fit=%.4f  (contrast=%.3f  reg=%.3f  period=%.3f  consist=%.3f)\n",
           ind.window_size, ind.ez_factor, ind.min_motif_count,
           fs.composite, fs.contrast, fs.spacing_regularity, fs.period_reward, fs.spacing_consistency);
}
