#pragma once
// =============================================================================
//  surrogate.cuh -- null-model series for contrast correction
// =============================================================================
//  A surrogate keeps some statistical property of the input series and
//  destroys the repeating structure.  Fitness signals computed on it give the
//  value a window size gets "for free", which evaluate_fitness subtracts.
//
//    shuffle : random permutation of the values.  Keeps the value
//              distribution; destroys all temporal structure (spectrum too).
//    phase   : Fourier phase randomisation.  Keeps the power spectrum
//              (resonances, periodic tones); destroys phase coupling, so
//              impulse shapes and amplitude modulation (the envelope that
//              carries a bearing fault frequency) are gone.
//
//  Both are deterministic for a given seed, so every GA individual, every
//  multi-seed repeat and the landscape sweep score against the same null.
// =============================================================================

#include <vector>

enum class SurrogateKind { Shuffle, Phase };

// Returns false (and leaves out untouched) for an unknown name.
bool parse_surrogate_kind(const char *name, SurrogateKind *out);
const char *surrogate_kind_name(SurrogateKind k);

// out must hold n floats.  Phase mode uses an O(n^2) real DFT on the host
// (~0.1 s at n = 6000); it prints a warning above n = 50000.
void make_surrogate(const float *h_T, int n, SurrogateKind kind,
                    unsigned int seed, float *out);
