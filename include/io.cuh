#pragma once
// =============================================================================
//  io.cuh -- Time series file I/O and Matrix Profile output
//
//  Loaders:
//    load_series_csv   -- one float per line, or two columns (index, value)
//    load_series_bin   -- raw little-endian float32 binary blob
//
//  Savers:
//    save_mp_csv       -- writes index, mp_distance, mp_index per row
//    save_series_csv   -- dumps the (possibly synthetic) series for inspection
// =============================================================================

#include "common.cuh"

// -----------------------------------------------------------------------------
//  Load a time series from a CSV file.
//  Supports two formats:
//    single-column : one float per line
//    two-column    : "timestamp,value" -- only the value column is read
//  Comment lines starting with '#' are skipped.
//  Returns a heap-allocated float array; sets *n_out to the number of samples.
//  Returns NULL on failure.
// -----------------------------------------------------------------------------
float* load_series_csv(const char* path, int* n_out);

// -----------------------------------------------------------------------------
//  Load a time series from a raw binary file (IEEE 754 float32, little-endian).
//  File size must be a multiple of 4 bytes.
//  Returns NULL on failure.
// -----------------------------------------------------------------------------
float* load_series_bin(const char* path, int* n_out);

// -----------------------------------------------------------------------------
//  Save the Matrix Profile and its index array to a CSV file.
//  Output format (header + one row per subsequence):
//    subseq_index, mp_distance, mp_index
//  mp_index == -1 means the index was not recovered (GA evaluation mode).
// -----------------------------------------------------------------------------
void save_mp_csv(
    const char*  path,
    const float* mp,
    const int*   mpi,
    int          profile_len,
    int          window_size   // written as a comment in the header
);

// -----------------------------------------------------------------------------
//  Save a float array as a single-column CSV (useful for exporting the
//  synthetic series or fitness history for external plotting).
// -----------------------------------------------------------------------------
void save_series_csv(const char* path, const float* data, int n);
