// =============================================================================
//  surrogate.cu -- shuffle and phase-randomised surrogates (host only)
// =============================================================================
#include "surrogate.cuh"
#include "common.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>

bool parse_surrogate_kind(const char *name, SurrogateKind *out)
{
    if (!strcmp(name, "shuffle")) { *out = SurrogateKind::Shuffle; return true; }
    if (!strcmp(name, "phase"))   { *out = SurrogateKind::Phase;   return true; }
    return false;
}

const char *surrogate_kind_name(SurrogateKind k)
{
    return k == SurrogateKind::Phase ? "phase" : "shuffle";
}

// -----------------------------------------------------------------------------
//  Phase randomisation
//    X_k = sum_j x_j exp(-2 pi i k j / n),  k = 0 .. n/2
//    Y_k = |X_k| exp(i phi_k),  phi_k ~ U[0, 2 pi)  for 0 < k < n/2
//    Y_0 = X_0 (mean kept);  Y_{n/2} = +-|X_{n/2}| for even n (must be real)
//    y_j = (1/n) [ Y_0 + 2 sum_{0<k<n/2} Re(Y_k e^{+2 pi i k j/n}) + Y_{n/2} (-1)^j ]
//  Index (k*j) mod n is advanced incrementally against a twiddle table, so the
//  whole thing is two O(n^2) multiply-add passes with no trig in the loop.
// -----------------------------------------------------------------------------
static void phase_surrogate(const float *x, int n, std::mt19937 &rng, float *out)
{
    if (n > 50000)
        fprintf(stderr, "[surrogate] n = %d: O(n^2) DFT will be slow\n", n);

    std::vector<double> cw(n), sw(n);
    for (int t = 0; t < n; t++) {
        double a = 2.0 * M_PI * (double)t / (double)n;
        cw[t] = cos(a);
        sw[t] = sin(a);
    }

    const int half = n / 2;
    std::vector<double> re(half + 1, 0.0), im(half + 1, 0.0);
    for (int k = 0; k <= half; k++) {
        double sr = 0.0, si = 0.0;
        long long idx = 0;
        for (int j = 0; j < n; j++) {
            sr += x[j] * cw[idx];
            si -= x[j] * sw[idx];
            idx += k;
            if (idx >= n) idx -= n;
        }
        re[k] = sr;
        im[k] = si;
    }

    std::uniform_real_distribution<double> U(0.0, 2.0 * M_PI);
    for (int k = 1; k <= half; k++) {
        double mag = sqrt(re[k] * re[k] + im[k] * im[k]);
        if (k == half && (n % 2 == 0)) {         // Nyquist bin must stay real
            re[k] = (U(rng) < M_PI) ? mag : -mag;
            im[k] = 0.0;
        } else {
            double phi = U(rng);
            re[k] = mag * cos(phi);
            im[k] = mag * sin(phi);
        }
    }

    const int last_full = (n % 2 == 0) ? half - 1 : half;   // bins counted twice
    for (int j = 0; j < n; j++) {
        double acc = re[0];
        long long idx = 0;
        for (int k = 1; k <= last_full; k++) {
            idx += j;
            idx %= n;
            acc += 2.0 * (re[k] * cw[idx] - im[k] * sw[idx]);
        }
        if (n % 2 == 0) acc += re[half] * ((j & 1) ? -1.0 : 1.0);
        out[j] = (float)(acc / n);
    }
}

void make_surrogate(const float *h_T, int n, SurrogateKind kind,
                    unsigned int seed, float *out)
{
    std::mt19937 rng(seed);
    if (kind == SurrogateKind::Shuffle) {
        std::copy(h_T, h_T + n, out);
        std::shuffle(out, out + n, rng);
    } else {
        phase_surrogate(h_T, n, rng, out);
    }
}
