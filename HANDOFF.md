# GA-STOMP Handoff Document

## Project Status: Non-overlapping Gap Fix Applied (v6-spacing-regularity)

### What Changed
**Single file change: `src/fitness.cu`**
- `compute_spacing_regularity()` now filters gaps: only counts gaps ≥ window_size (`w`)
- Prevents adjacent positions from the same motif instance inflating regularity
- Added minimum 4 non-overlapping gaps requirement (else regularity = 0)
- Call site passes `ind.window_size`

### Build Command (Kaggle P100, sm_60)
```bash
nvcc -std=c++17 -arch=sm_60 -Iinclude -O3 -DNDEBUG \
  --expt-relaxed-constexpr -o ga_stomp \
  src/main.cu src/stomp.cu src/utils.cu src/fitness.cu \
  src/ga.cu src/io.cu src/timer.cu src/config.cu src/benchmark.cu
```

### Run Config (per dataset)
```ini
m_min=20
m_max=200
ez_min=0.25
ez_max=1.0
k_min=2
k_max=20
population=20
generations=20
tournament_k=3
mutation_rate=0.30
elite_count=2
approx_frac=0.5
n_seeds=3
verbose=1
input_path=data/IR007_1797.csv   # change per dataset
output_dir=results/IR007         # change per dataset
```

### Datasets & Expected Results

| Dataset | File | true_m | Acceptable Range (±20%) | Seeds |
|---------|------|--------|-------------------------|-------|
| IR007   | 105.mat (X105_DE_time) | 74 | [59, 89] | 42, 1042, 2042 |
| IR014   | 106.mat (X106_DE_time) | 74 | [59, 89] | 42, 1042, 2042 |
| OR007   | 130.mat (X130_DE_time) | 112 | [90, 134] | 42, 1042, 2042 |
| Normal  | 100.mat (X100_DE_time) | N/A | N/A (no defect) | 42, 1042, 2042 |

**Signal:** First 6000 samples, 12kHz, 1797 RPM (bearing defect frequencies known)

### Pass Criterion
✅ **PASS** = best_window_size within ±20% of true_m for **all 3 seeds** on IR007, IR014, OR007
❌ **FAIL** = any seed outside range, or systematic drift to m_max (187-188)

### Expected Multi-Seed Output Table (per dataset)

```
+----------+------------+------------+------------+----------+----------+
| Dataset  | Seed 42    | Seed 1042  | Seed 2042  | Mean ±SD | In Range?|
+----------+------------+------------+------------+----------+----------+
| IR007    |     74     |     76     |     72     | 74.0±2.0 |   YES    |
| IR014    |     73     |     75     |     74     | 74.0±1.0 |   YES    |
| OR007    |    112     |    110     |    114     | 112.0±2.0|   YES    |
| Normal   |    N/A     |    N/A     |    N/A     |   N/A    |   N/A    |
+----------+------------+------------+------------+----------+----------+
```

### If Fix Fails (Reward Hacking Persists)

**Fallback: FFT Reference-Period Approach** (validated on synthetic)
- Compute MP once at fixed `w_ref=30` before GA
- FFT of `(mean_mp - mp)`, find dominant period in [30, 400]
- Score candidate `m` with Gaussian reward centered on detected period
- Files to change: `fitness.cuh`, `fitness.cu`, `ga.cu` (3 files)
- Detected periods: IR007→74, OR007→112 (synthetic validation)

### History of Failed Attempts (for context)
| Attempt | Problem | Why It Failed |
|---------|---------|---------------|
| count_score in composite | GA gamed k to match discovered | count_score = 1.0 always achievable |
| consistency signal | Self-referential bias | Used same MP positions for selection & scoring |
| GAP_LO floor tuning | Arbitrary threshold | Didn't address root cause (overlapping gaps) |
| Sigmoid prior on m | Biased search | Masked fitness landscape issues |
| Fixed-count (50) selection | Large m + large ez → fake uniform spacing | Exclusion zone forced geometric uniformity |

### Key Insight
The reward hack: **large m + large ez_factor** forces the 50 selected positions to be spaced by the exclusion zone, creating artificially low CV (high regularity). The non-overlapping gap filter breaks this by ignoring gaps < window_size.

### Files to Monitor
- `src/fitness.cu` — fix location (lines 77-147)
- `results/*/multiseed.csv` — reproducibility report
- `results/*/sweep.csv` — fitness-recall correlation (Spearman ρ)

### Next Steps After Kaggle Run
1. If PASS: Document results, consider paper submission
2. If FAIL: Implement FFT reference-period fallback (3-file change)
3. Either way: Update this HANDOFF with actual results
