// =============================================================================
//  io.cu -- Time series file I/O and Matrix Profile output
// =============================================================================

#include "io.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <vector>

// -----------------------------------------------------------------------------
//  load_series_csv
// -----------------------------------------------------------------------------
//  Accepts two formats:
//    single-column : "3.14"          -> value = 3.14
//    two-column    : "0,3.14"        -> value = second token
//  Lines starting with '#' are treated as comments and skipped.
//  Blank lines are skipped.
float* load_series_csv(const char* path, int* n_out)
{
    FILE* fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "[io] Cannot open '%s': %s\n", path, strerror(errno));
        *n_out = 0;
        return NULL;
    }

    std::vector<float> vals;
    char line[256];

    while (fgets(line, sizeof(line), fp)) {
        // Skip comment and blank lines
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;

        // Try two-column format first
        float a, b;
        if (sscanf(line, "%f,%f", &a, &b) == 2) {
            vals.push_back(b);
        } else if (sscanf(line, "%f", &a) == 1) {
            vals.push_back(a);
        }
        // Silently skip unparseable lines
    }
    fclose(fp);

    if (vals.empty()) {
        fprintf(stderr, "[io] No numeric data found in '%s'\n", path);
        *n_out = 0;
        return NULL;
    }

    *n_out = (int)vals.size();
    float* out = (float*)malloc(vals.size() * sizeof(float));
    memcpy(out, vals.data(), vals.size() * sizeof(float));

    printf("[io] Loaded %d samples from '%s'\n", *n_out, path);
    return out;
}

// -----------------------------------------------------------------------------
//  load_series_bin
// -----------------------------------------------------------------------------
float* load_series_bin(const char* path, int* n_out)
{
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "[io] Cannot open '%s': %s\n", path, strerror(errno));
        *n_out = 0;
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    long bytes = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    if (bytes % sizeof(float) != 0) {
        fprintf(stderr, "[io] '%s' size (%ld bytes) not divisible by 4\n",
                path, bytes);
        fclose(fp);
        *n_out = 0;
        return NULL;
    }

    int n = (int)(bytes / sizeof(float));
    float* out = (float*)malloc(bytes);
    if ((int)fread(out, sizeof(float), n, fp) != n) {
        fprintf(stderr, "[io] Read error on '%s'\n", path);
        free(out);
        fclose(fp);
        *n_out = 0;
        return NULL;
    }
    fclose(fp);

    *n_out = n;
    printf("[io] Loaded %d samples from binary '%s'\n", n, path);
    return out;
}

// -----------------------------------------------------------------------------
//  save_mp_csv
// -----------------------------------------------------------------------------
void save_mp_csv(
    const char*  path,
    const float* mp,
    const int*   mpi,
    int          profile_len,
    int          window_size)
{
    FILE* fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr, "[io] Cannot write '%s': %s\n", path, strerror(errno));
        return;
    }

    fprintf(fp, "# Matrix Profile output\n");
    fprintf(fp, "# window_size = %d\n", window_size);
    fprintf(fp, "# profile_len = %d\n", profile_len);
    fprintf(fp, "subseq_index,mp_distance,mp_index\n");

    for (int i = 0; i < profile_len; i++) {
        fprintf(fp, "%d,%.6f,%d\n", i, mp[i], mpi ? mpi[i] : -1);
    }

    fclose(fp);
    printf("[io] Saved MP (%d rows) -> '%s'\n", profile_len, path);
}

// -----------------------------------------------------------------------------
//  save_series_csv
// -----------------------------------------------------------------------------
void save_series_csv(const char* path, const float* data, int n)
{
    FILE* fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr, "[io] Cannot write '%s': %s\n", path, strerror(errno));
        return;
    }

    fprintf(fp, "# time series export  n=%d\n", n);
    fprintf(fp, "index,value\n");
    for (int i = 0; i < n; i++) {
        fprintf(fp, "%d,%.6f\n", i, data[i]);
    }

    fclose(fp);
    printf("[io] Saved series (%d samples) -> '%s'\n", n, path);
}
