# GA + GPU-STOMP — Agent Quick Reference

## Build & Run
```bash
# Edit SM_ARCH in Makefile for your GPU (default sm_86 = RTX 30xx/A100)
make            # release (-O3)
make debug      # debug (-G -g)
make profile    # Nsight (-lineinfo -pg)
./ga_stomp                      # synthetic defaults (n=4000, m=80, pop=30, gen=20)
./ga_stomp config.ini           # load from INI file
./ga_stomp 8000 120 50 30       # quick synthetic: n true_m pop gens
./ga_stomp --help               # usage
```

### Kaggle P100 (sm_60) Build
```bash
nvcc -std=c++17 -arch=sm_60 -Iinclude -O3 -DNDEBUG \
  --expt-relaxed-constexpr -o ga_stomp \
  src/main.cu src/stomp.cu src/utils.cu src/fitness.cu \
  src/ga.cu src/io.cu src/timer.cu src/config.cu src/benchmark.cu
```

## Key Config Keys (sample_config.ini)
| Key | Default | Notes |
|-----|---------|-------|
| `m_min` / `m_max` | 10 / 300 | window size bounds |
| `ez_min` / `ez_max` | 0.25 / 1.0 | exclusion zone factor |
| `k_min` / `k_max` | 2 / 20 | min motif count |
| `population` | 50 | GA population size |
| `generations` | 30 | GA generations |
| `approx_frac` | 1.0 | <1.0 = SCRIMP++ approximation (faster, lower bound) |
| `n_seeds` | 5 | multi-seed reproducibility runs |
| `input_path` | (blank) | blank = synthetic; .csv/.bin = real data |
| `output_dir` | . | all CSV outputs go here |

### Kaggle Run Config
```ini
m_min=20 m_max=200 ez_min=0.25 ez_max=1.0 k_min=2 k_max=20
population=20 generations=20 tournament_k=3 mutation_rate=0.30 elite_count=2
approx_frac=0.5 n_seeds=3 verbose=1
input_path=data/IR007_1797.csv
output_dir=results/IR007
```

## Architecture Notes
- **Single binary** (`ga_stomp`) — no libraries, no install step
- **CUDA 11+**, compute capability ≥ 7.5 (Turing) — **Kaggle P100 uses sm_60 (CUDA 11/12)**
- **Diagonal parallelism**: each CUDA thread processes one diagonal of the distance matrix
- **GA evaluates full population per generation** — maps naturally to CUDA batch execution
- **Fitness = 0.35×contrast + 0.65×regularity** (count_score excluded — gameable)
- **Approximation mode** (`approx_frac < 1.0`) processes strided diagonals; produces valid lower bound on MP

## Output Files (written to `output_dir/`)
| File | Purpose |
|------|---------|
| `series.csv` | input time series (copy) |
| `matrix_profile.csv` | final MP distances + indices |
| `timing.csv` | kernel/GA/fitness timing breakdown |
| `multiseed.csv` | mean±std over N seeds |
| `sweep.csv` | fitness-recall sweep (Spearman ρ) |
| `run_config.ini` | exact config used (reproducibility) |

## Common Pitfalls
- **SM_ARCH mismatch** → kernel launch fails; edit Makefile line 17 (or `-arch=sm_60` for P100)
- **`approx_frac` during GA** speeds search but final run always uses 1.0
- **Real data**: set `input_path` in config; `true_m` unknown → sweep skipped
- **MPI recovery** (`recover_indices=true`) only for final run; adds CPU pass
- **Fitness landscape is jagged** — GA > grid search > Bayesian opt for this problem

## Extending the Framework
The GA logic (`ga.cu`) is model-agnostic. To adapt:
1. Change `Individual` struct genes
2. Replace `run_stomp` in `evaluate_individual` with your GPU evaluator
3. Replace `evaluate_fitness` with your quality metric
4. Crossover/mutation/selection need **no changes**

## Validation Strategy (from README)
1. **Synthetic benchmark** — inject known motif, verify GA finds `m` within ±5%
2. **Fitness-recall correlation** — Spearman ρ > 0.6 validates fitness proxy
3. **Cross-dataset** — ECG, industrial sensor, audio without retuning

## Reproducibility
- All RNG seeded (base seed 42, derived seeds for multi-seed)
- `run_config.ini` saved automatically
- Elitism preserves best-found solution

## Profiling
```bash
make profile
nsys profile --stat=true ./ga_stomp 4000 80 50 30
# Check: stomp_diagonal_kernel occupancy, atomic contention, H2D/D2H overhead
```

---

## Kaggle CWRU Status: v6 Fix FAILED — Implementing FFT Fallback

### Latest Run Results (v6 Non-overlapping Gap Fix)
| Dataset | true_m | Seed 42 | Seed 1042 | Seed 2042 | Mean ± SD | PASS? |
|---------|--------|---------|-----------|-----------|-----------|-------|
| IR007   | 74     | 188     | 128       | 21        | 112.3±69.1 | ❌ |
| IR014   | 74     | 187     | 23        | 23        | 77.7±77.3  | ❌ |
| OR007   | 112    | 103     | 20        | 20        | 47.7±39.1  | ❌ |
| Normal  | N/A    | 38      | 20        | 20        | 26.0±8.5   | N/A |

**Root cause**: Reward hacking persists. Large `m` + large `ez_factor` forces exclusion-zone-spaced positions → artificially low CV (high regularity). Non-overlapping gap filter insufficient.

### Next Step: FFT Reference-Period Approach (Active)
**Implementation required** (3 files: `fitness.cuh`, `fitness.cu`, `ga.cu`):
1. Compute MP once at fixed `w_ref=30` before GA (in `ga.cu`)
2. FFT of `(mean_mp - mp)`, find dominant period in [30, 400] (in `fitness.cu`)
3. Score candidate `m` with Gaussian reward centered on detected period (in `fitness.cu`)
4. Pass detected period to GA via config/Individual

**Validated on synthetic**: detects period=74 (IR007) and period=112 (OR007)

See `HANDOFF.md` for full history and implementation details.
