// =============================================================================
//  config.cu -- Key=value configuration file implementation
// =============================================================================

#include "config.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>

// Trim leading/trailing whitespace in-place, return pointer to first non-space
static char* trim(char* s)
{
    while (isspace((unsigned char)*s)) s++;
    if (*s == '\0') return s;
    char* end = s + strlen(s) - 1;
    while (end > s && isspace((unsigned char)*end)) end--;
    *(end + 1) = '\0';
    return s;
}

// -----------------------------------------------------------------------------
//  Default config
// -----------------------------------------------------------------------------
FullConfig default_full_config()
{
    FullConfig fc;
    fc.ga.m_min           = 10;
    fc.ga.m_max           = 300;
    fc.ga.ez_min          = 0.25f;
    fc.ga.ez_max          = 1.0f;
    fc.ga.k_min           = 2;
    fc.ga.k_max           = 20;
    fc.ga.population_size = 50;
    fc.ga.generations     = 30;
    fc.ga.tournament_k    = 3;
    fc.ga.mutation_rate   = 0.30f;
    fc.ga.elite_count     = 2;
    fc.ga.approx_frac     = 1.0f;
    fc.ga.verbose         = true;
    fc.approx_frac        = 1.0f;
    fc.n_seeds            = 5;
    fc.input_path[0]      = '\0';
    fc.output_dir[0]      = '.';
    fc.output_dir[1]      = '\0';
    return fc;
}

// -----------------------------------------------------------------------------
//  load_config
// -----------------------------------------------------------------------------
bool load_config(const char* path, FullConfig* out)
{
    *out = default_full_config();

    FILE* fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "[config] Cannot open '%s'\n", path);
        return false;
    }

    char line[512];
    while (fgets(line, sizeof(line), fp)) {
        char* p = trim(line);
        if (*p == '#' || *p == '\0') continue;   // comment or blank

        // Split on '='
        char* eq = strchr(p, '=');
        if (!eq) continue;
        *eq = '\0';
        char* key = trim(p);
        char* val = trim(eq + 1);

        // Strip inline comment from value
        char* hash = strchr(val, '#');
        if (hash) { *hash = '\0'; val = trim(val); }

        // Match keys
        if      (!strcmp(key, "m_min"))         out->ga.m_min           = atoi(val);
        else if (!strcmp(key, "m_max"))         out->ga.m_max           = atoi(val);
        else if (!strcmp(key, "ez_min"))        out->ga.ez_min          = (float)atof(val);
        else if (!strcmp(key, "ez_max"))        out->ga.ez_max          = (float)atof(val);
        else if (!strcmp(key, "k_min"))         out->ga.k_min           = atoi(val);
        else if (!strcmp(key, "k_max"))         out->ga.k_max           = atoi(val);
        else if (!strcmp(key, "population"))    out->ga.population_size = atoi(val);
        else if (!strcmp(key, "generations"))   out->ga.generations     = atoi(val);
        else if (!strcmp(key, "tournament_k"))  out->ga.tournament_k    = atoi(val);
        else if (!strcmp(key, "mutation_rate")) out->ga.mutation_rate   = (float)atof(val);
        else if (!strcmp(key, "elite_count"))   out->ga.elite_count     = atoi(val);
        else if (!strcmp(key, "approx_frac")) {
            float f = (float)atof(val);
            out->approx_frac    = f;
            out->ga.approx_frac = f;
        }
        else if (!strcmp(key, "n_seeds"))     out->n_seeds  = atoi(val);
        else if (!strcmp(key, "verbose"))     out->ga.verbose = (atoi(val) != 0);
        else if (!strcmp(key, "input_path"))  strncpy(out->input_path,  val, 511);
        else if (!strcmp(key, "output_dir"))  strncpy(out->output_dir,  val, 511);
        else fprintf(stderr, "[config] Unknown key '%s' -- ignored\n", key);
    }
    fclose(fp);
    printf("[config] Loaded '%s'\n", path);
    return true;
}

// -----------------------------------------------------------------------------
//  save_config
// -----------------------------------------------------------------------------
void save_config(const char* path, const FullConfig& fc)
{
    FILE* fp = fopen(path, "w");
    if (!fp) { fprintf(stderr, "[config] Cannot write '%s'\n", path); return; }

    fprintf(fp, "# GA+STOMP configuration\n");
    fprintf(fp, "m_min         = %d\n",  fc.ga.m_min);
    fprintf(fp, "m_max         = %d\n",  fc.ga.m_max);
    fprintf(fp, "ez_min        = %.3f\n",fc.ga.ez_min);
    fprintf(fp, "ez_max        = %.3f\n",fc.ga.ez_max);
    fprintf(fp, "k_min         = %d\n",  fc.ga.k_min);
    fprintf(fp, "k_max         = %d\n",  fc.ga.k_max);
    fprintf(fp, "population    = %d\n",  fc.ga.population_size);
    fprintf(fp, "generations   = %d\n",  fc.ga.generations);
    fprintf(fp, "tournament_k  = %d\n",  fc.ga.tournament_k);
    fprintf(fp, "mutation_rate = %.3f\n",fc.ga.mutation_rate);
    fprintf(fp, "elite_count   = %d\n",  fc.ga.elite_count);
    fprintf(fp, "approx_frac   = %.3f\n",fc.approx_frac);
    fprintf(fp, "n_seeds       = %d\n",  fc.n_seeds);
    fprintf(fp, "verbose       = %d\n",  fc.ga.verbose ? 1 : 0);
    if (fc.input_path[0]) fprintf(fp, "input_path    = %s\n", fc.input_path);
    fprintf(fp, "output_dir    = %s\n",  fc.output_dir);
    fclose(fp);
    printf("[config] Saved config -> '%s'\n", path);
}

// -----------------------------------------------------------------------------
//  print_config
// -----------------------------------------------------------------------------
void print_config(const FullConfig& fc)
{
    printf("+--------------------------------------------------------------+\n");
    printf("|                  ACTIVE CONFIGURATION                       |\n");
    printf("+--------------------------------------------------------------+\n");
    printf("|  m range        : [%d, %d]\n", fc.ga.m_min, fc.ga.m_max);
    printf("|  ez range       : [%.2f, %.2f]\n", fc.ga.ez_min, fc.ga.ez_max);
    printf("|  k range        : [%d, %d]\n", fc.ga.k_min, fc.ga.k_max);
    printf("|  population     : %d\n", fc.ga.population_size);
    printf("|  generations    : %d\n", fc.ga.generations);
    printf("|  tournament_k   : %d\n", fc.ga.tournament_k);
    printf("|  mutation_rate  : %.3f\n", fc.ga.mutation_rate);
    printf("|  elite_count    : %d\n", fc.ga.elite_count);
    printf("|  approx_frac    : %.3f\n", fc.approx_frac);
    printf("|  n_seeds        : %d\n", fc.n_seeds);
    printf("|  input_path     : %s\n", fc.input_path[0] ? fc.input_path : "(synthetic)");
    printf("|  output_dir     : %s\n", fc.output_dir);
    printf("+--------------------------------------------------------------+\n\n");
}
