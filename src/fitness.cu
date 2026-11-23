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
//  Distinct motif instances (non-maximum suppression)
// =============================================================================
//  Walks MP positions in ascending distance order and accepts a position only
//  if it is at least `radius` samples from every position already accepted.
//  Neighbouring subsequences of one motif occurrence have almost identical MP
//  values, so without this step "the top-50 positions" or "the k-th smallest
//  value" are mostly trivial neighbours of the single best match.
//  Returns positions in acceptance (best-first) order.
// =============================================================================
static inline bool mp_valid(float v) { return std::isfinite(v) && v < 0.5f * INF_F; }

static std::vector<int> pick_instances(const float *mp, int L, int radius, int max_count)
{
    std::vector<int> order(L);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](int a, int b) { return mp[a] < mp[b]; });

    radius = std::max(radius, 1);
    std::vector<char> blocked(L, 0);
    std::vector<int>  out;
    out.reserve(max_count);
    for (int idx : order)
    {
        if (!mp_valid(mp[idx])) break;          // sorted: rest are INF too
        if (blocked[idx]) continue;
        out.push_back(idx);
        int lo = std::max(0, idx - radius + 1);
        int hi = std::min(L - 1, idx + radius - 1);
        for (int j = lo; j <= hi; j++) blocked[j] = 1;
        if ((int)out.size() >= max_count) break;
    }
    return out;
}

// =============================================================================
//  Signal 1 -- Motif Contrast
// =============================================================================
//    contrast = (d_k - d_top) / mean(MP)        score = 1 - exp(-contrast)
//
//  d_top = distance of the best motif instance
//  d_k   = distance of the k-th DISTINCT instance (instances >= m apart),
//          i.e. the background level k occurrences down.
//
//  Before: d_k was the k-th smallest raw MP value, which is almost always a
//  trivial neighbour of the top match, so contrast was ~0.01 on CWRU and the
//  GA pushed k to k_max to squeeze out a larger gap.
// =============================================================================
static float compute_contrast(
    const float *mp,
    int          L,
    int          w,
    int          min_motif_count,
    float        mean_mp)
{
    int k = std::max(1, min_motif_count);
    std::vector<int> inst = pick_instances(mp, L, w, k + 1);
    if ((int)inst.size() < 2) return 0.0f;

    float d_top = mp[inst.front()];
    float d_k   = mp[inst.back()];
    float contrast = (d_k - d_top) / (mean_mp + EPS);
    return 1.0f - expf(-fmaxf(contrast, 0.0f));
}

// =============================================================================
//  Signal 2 -- Spacing Regularity  (distinct instances, harmonic fit)
// =============================================================================
//  1. Pick K distinct motif instances with non-maximum suppression at radius
//     w (so every gap is >= w by construction):
//        K = clamp(L / (2w), 6, 30)
//  2. Gaps g_i between consecutive instances (sorted by position).
//  3. Base spacing g0 = median gap.  Each gap should be close to an integer
//     multiple of g0 (a skipped occurrence gives 2*g0, which is still regular):
//        r_i = | g_i/g0 - max(1, round(g_i/g0)) |      r_i in [0, 0.5]
//  4. regularity = clamp(1 - 4 * mean(r_i), 0, 1)
//        perfectly periodic -> 1;  uniformly random spacing (E[r] = 0.25) -> 0
//
//  Why the old version returned 0 on CWRU: it took the 50 lowest raw MP
//  positions, which cluster around two or three occurrences, then kept only
//  gaps >= w.  Fewer than 4 gaps survived, so it hit the early return.
//
//  Known limitation (also visible in landscape.csv): once w spans a full
//  period every phase of the cycle matches equally well, so instance positions
//  stop being phase-locked and regularity falls.  This signal therefore tends
//  to favour windows around the length of one event, not the event period.
// =============================================================================
static void compute_spacing_regularity(
    const float *mp,
    int          L,
    int          w,
    float       &regularity_out)
{
    regularity_out = 0.0f;
    w = std::max(w, 1);
    int K = std::min(30, std::max(6, L / (2 * w)));
    std::vector<int> inst = pick_instances(mp, L, w, K);
    if ((int)inst.size() < 6) return;

    std::sort(inst.begin(), inst.end());
    std::vector<float> gaps;
    gaps.reserve(inst.size() - 1);
    for (size_t i = 1; i < inst.size(); i++)
        gaps.push_back((float)(inst[i] - inst[i - 1]));

    std::vector<float> tmp(gaps);
    std::sort(tmp.begin(), tmp.end());
    size_t h = tmp.size() / 2;
    float g0 = (tmp.size() % 2) ? tmp[h] : 0.5f * (tmp[h - 1] + tmp[h]);
    if (g0 < 1.0f) return;

    float r_sum = 0.0f;
    for (float g : gaps)
    {
        float q = g / g0;
        float n = fmaxf(1.0f, roundf(q));
        r_sum += fabsf(q - n);
    }
    float r_mean = r_sum / (float)gaps.size();
    regularity_out = fminf(1.0f, fmaxf(0.0f, 1.0f - 4.0f * r_mean));
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
    const float         *mp,
    int                  profile_len,
    const Individual    &ind,
    int                  detected_period,   // -1 = no period reward
    const FitnessWeights &w,
    const float         *null_mp)
{
    FitnessScore fs{};
    fs.contrast_null   = NAN;
    fs.regularity_null = NAN;

    if (profile_len <= 0 || mp == nullptr)
    {
        fs.composite = -1.0f;
        return fs;
    }

    // Statistics over valid (non-INF) MP entries only; approximate mode can
    // leave rows uncovered.
    float mp_min = INF_F, mp_max = -INF_F, mean_mp = 0.0f;
    int   n_valid = 0;
    for (int i = 0; i < profile_len; i++)
    {
        if (!mp_valid(mp[i])) continue;
        mp_min = fminf(mp_min, mp[i]);
        mp_max = fmaxf(mp_max, mp[i]);
        mean_mp += mp[i];
        n_valid++;
    }
    if (n_valid < 2 || mp_max < EPS || (mp_max - mp_min) < EPS)
    {
        fs.composite = 0.0f;               // degenerate profile
        return fs;
    }
    mean_mp /= (float)n_valid;

    // --- Signals -------------------------------------------------------------
    fs.contrast_raw = compute_contrast(mp, profile_len, ind.window_size,
                                       ind.min_motif_count, mean_mp);
    fs.contrast = fs.contrast_raw;

    // Null correction: subtract what the same window gets on a surrogate with
    // the repeating structure destroyed.  Signed (can be < 0) so the GA still
    // has a gradient where real and surrogate are close.
    if (null_mp != nullptr)
    {
        float null_mean = 0.0f;
        int   null_valid = 0;
        for (int i = 0; i < profile_len; i++)
            if (mp_valid(null_mp[i])) { null_mean += null_mp[i]; null_valid++; }
        if (null_valid > 1)
        {
            null_mean /= (float)null_valid;
            fs.contrast_null = compute_contrast(null_mp, profile_len, ind.window_size,
                                                ind.min_motif_count, null_mean);
            compute_spacing_regularity(null_mp, profile_len, ind.window_size,
                                       fs.regularity_null);
            fs.contrast = fs.contrast_raw - fs.contrast_null;
        }
    }
    compute_spacing_regularity(mp, profile_len, ind.window_size, fs.spacing_regularity);

    if (detected_period > 0)
    {
        float sigma = 0.20f * (float)detected_period;   // +-40% keeps ~95% of mass
        fs.period_reward = compute_period_reward(ind.window_size, detected_period, sigma);
    }
    else
    {
        fs.period_reward = 0.5f;                         // neutral
    }

    // Legacy: logged for ablation only, never in the composite
    fs.count_score = compute_count_score(mp, profile_len, mp_min, mean_mp,
                                         ind.min_motif_count, fs.discovered_motifs);
    fs.spacing_consistency = fs.spacing_regularity;

    // --- Composite -------------------------------------------------------------
    //   Defaults (FitnessWeights): 0.20 contrast + 0.30 regularity + 0.50 period.
    //   Weights are configurable (w_contrast / w_regularity / w_period) for
    //   ablations; they need not sum to 1 since the GA only ranks individuals.
    fs.composite = w.contrast   * fs.contrast
                 + w.regularity * fs.spacing_regularity
                 + w.period     * fs.period_reward;
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
