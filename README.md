# GA + GPU-STOMP: Evolutionary Hyperparameter Search for Time Series Motif Discovery

A CUDA-accelerated framework that uses a Genetic Algorithm (GA) to jointly
optimise the three key hyperparameters of STOMP-based motif discovery, replacing
the standard practice of manually tuning a single fixed window size.

---

## Table of Contents

1. [Motivation](#1-motivation)
2. [Algorithm Overview](#2-algorithm-overview)
3. [Individual Encoding](#3-individual-encoding)
4. [STOMP — GPU Implementation](#4-stomp--gpu-implementation)
5. [Memory Hierarchy and Optimisations](#5-memory-hierarchy-and-optimisations)
6. [Fitness Function](#6-fitness-function)
7. [Genetic Algorithm](#7-genetic-algorithm)
8. [Comparison to Prior Methods](#8-comparison-to-prior-methods)
9. [Project Structure](#9-project-structure)
10. [Build and Run](#10-build-and-run)
11. [Extending to Other Models](#11-extending-to-other-models)
12. [Complexity Analysis](#12-complexity-analysis)
13. [Validation Strategy](#13-validation-strategy)
14. [Conference Q&A Reference](#14-conference-qa-reference)
15. [References](#15-references)

---

## 1. Motivation

The Matrix Profile (MP) and its STOMP algorithm are the state-of-the-art in
time series motif discovery. However, every STOMP run requires a fixed window
size `m`, which critically affects result quality:

| m too small | m too large |
|---|---|
| Noise-level matches dominate | All subsequences look alike |
| Trivial (overlapping) motifs | Meaningful patterns are over-smoothed |
| MP concentrated near zero | MP is flat with no contrast |

The standard advice is "try several m values manually." This project replaces
manual search with a principled evolutionary search over the joint space
`[m, exclusion_zone_factor, min_motif_count]`, using CUDA to make the
otherwise prohibitive computational cost tractable.

---

## 2. Algorithm Overview

```
Initialise random population of N individuals
  │
  ▼  (repeat for G generations)
┌──────────────────────────────────────────────────────────┐
│  For each individual [m, ez, k]:                         │
│    ① Upload time series T to GPU                         │
│    ② compute_stats_kernel  → means[], stds[]             │
│    ③ compute_qt_init_kernel → QT_init[]  (diagonal seeds)│
│    ④ stomp_diagonal_kernel  → MP[]       (matrix profile)│
│    ⑤ Download MP to CPU                                  │
│    ⑥ evaluate_fitness(MP) → composite score              │
└──────────────────────────────────────────────────────────┘
  │
  ▼
Select parents via k-tournament
Crossover  → arithmetic blend (m, ez) + random-inherit (k)
Mutate     → Gaussian noise with temperature annealing
Elitism    → top-2 survive unchanged
  │
  ▼
Return best individual and its final Matrix Profile
```

CUDA provides parallelism **within** each STOMP evaluation (across diagonals).
The GA provides direction **across** evaluations (exploring hyperparameter space).

---

## 3. Individual Encoding

Each GA individual encodes three genes:

```
Individual {
    window_size:         int    ∈ [m_min, m_max]      (default 10–300)
    ez_factor:           float  ∈ [0.25,  1.0]
    min_motif_count:     int    ∈ [2,     20]
}
```

`exclusion_zone = floor(ez_factor × m)`

The exclusion zone prevents trivial self-matches. The standard STOMP heuristic
is `ez_factor = 0.25`, but this project treats it as a learnable parameter
because the optimal exclusion zone depends on the dominant periodicity of the
signal, which interacts with `m` in a non-trivial way.

`min_motif_count` guides the fitness function's count signal but does not enter
the STOMP kernel — it parameterises the "how many motifs is ideal" expectation.

---

## 4. STOMP — GPU Implementation

### 4.1 Why Diagonal Parallelisation?

STOMP's inner engine is a sliding dot-product update (the QT recurrence):

```
QT(i, j) = QT(i-1, j-1) − T[i−1]·T[j−1] + T[i+m−1]·T[j+m−1]
```

This means **row `i+1` depends on row `i`** → horizontal (row-wise)
parallelism collapses into a sequential chain.

**Diagonals** have no inter-diagonal dependency. Diagonal `d` represents all
pairs `(i, i+d)` for `i = 0 … L−d−1` where `L = n−m+1`.  These pairs are
completely independent of diagonal `d+1`, making the problem
**embarrassingly parallel across diagonals**.

```
Distance matrix structure:

     j=0  j=1  j=2  j=3  j=4
i=0 [  ─   d1   d2   d3   d4 ]
i=1 [ d1   ─    d1   d2   d3 ]
i=2 [ d2   d1   ─    d1   d2 ]
i=3 [ d3   d2   d1   ─    d1 ]

Columns of same shade = one diagonal = one CUDA thread
```

### 4.2 Kernel Mapping

```
Thread index = diagonal index d
  Thread 0 → diagonal ez+0  (L−ez−0 iterations)
  Thread 1 → diagonal ez+1  (L−ez−1 iterations)
  ...
  Thread k → diagonal ez+k
```

Each thread:
1. Loads `QT_init[d−ez]` (seed computed by the pre-kernel)
2. Slides along the diagonal updating `qt` via the recurrence
3. Converts `qt` to z-normalised Euclidean distance at each step
4. Calls `atomicMinFloat(&MP[i], dist)` and `atomicMinFloat(&MP[j], dist)`

### 4.3 Z-Normalised Distance Formula

```
Pearson(i, j)  =  (QT(i,j)/m  −  μᵢ·μⱼ)  /  (σᵢ·σⱼ)
dist_z(i, j)   =  √(2m·(1 − Pearson(i,j)))
```

Pearson is clamped to [−1, 1] before the square root to handle floating-point
rounding near ±1.

### 4.4 Pre-computation Kernels

**`compute_stats_kernel`** — One thread per subsequence.
Each thread computes mean and std of its `m` elements in a single pass using
`E[X²] − E[X]²`. Reads `T[i..i+m−1]` which are coalesced across threads.

**`compute_qt_init_kernel`** — One thread per diagonal.
Thread for diagonal `d` computes:
`QT_init[d−ez] = Σ_{k=0}^{m−1} T[k]·T[k+d]`
This is an O(nm) operation whose cost is negligible relative to O(n²) STOMP.

---

## 5. Memory Hierarchy and Optimisations

### 5.1 Memory Placement

```
┌────────────────────┬─────────────┬─────────────────────────────────────────┐
│ Variable           │ Memory      │ Rationale                               │
├────────────────────┼─────────────┼─────────────────────────────────────────┤
│ T[]                │ Global      │ Too large; read-only → __ldg cache      │
│ means[], stds[]    │ Global      │ Read-only per kernel → __ldg cache      │
│ QT_init[]          │ Global      │ Read once per diagonal → __ldg          │
│ MP[], MPI[]        │ Global      │ Written atomically; cannot cache        │
├────────────────────┼─────────────┼─────────────────────────────────────────┤
│ qt accumulator     │ Register    │ Updated every iteration — zero latency  │
│ mu_i, sig_i, etc.  │ Register    │ Loaded once from global, reused in loop │
│ dist, i, j, k      │ Register    │ Short-lived temporaries                 │
└────────────────────┴─────────────┴─────────────────────────────────────────┘
```

### 5.2 Applied Optimisations

**`__ldg` for read-only arrays**
All reads from `T`, `means`, `stds`, `QT_init` use `__ldg()`, routing through
the 48 KB read-only data cache (separate from L1). This doubles effective cache
capacity since MP writes do not evict T from the RO cache.

**`const __restrict__` pointer qualifiers**
Tell the compiler there is no aliasing between input and output pointers,
enabling automatic `__ldg` promotion and loop vectorisation.

**`atomicMinFloat` via int CAS**
CUDA has no native `atomicMin(float*, float)`. We reinterpret float bits as int
and use `atomicCAS`. Correct for all non-negative values (distances ≥ 0) because
IEEE 754 positive floats preserve order under integer interpretation.

```cuda
__device__ void atomicMinFloat(float* addr, float val) {
    int* addr_as_int = reinterpret_cast<int*>(addr);
    int  old  = *addr_as_int;
    int  bits = __float_as_int(val);
    while (val < __int_as_float(old)) {
        int assumed = old;
        old = atomicCAS(addr_as_int, assumed, bits);
        if (old == assumed) break;
    }
}
```

**Register accumulation of `qt`**
The QT update and distance computation operate entirely in registers.
The inner loop touches zero global memory addresses per iteration
(all T reads hit the RO cache; MP updates are atomic writes).

**Warp-level min reduction with `__shfl_down_sync`**
Available as a post-processing step to reduce atomic contention when multiple
diagonals update the same MP[i] simultaneously:

```cuda
float warp_min = warpReduceMin(local_dist);
if (lane_id == 0) atomicMinFloat(&MP[i], warp_min);
```

**Memory coalescing**
Thread `t` in block `b` handles diagonal `ez + b·BLOCK_SIZE + t`.
Reads of `T[i]` and `T[i+d]` for consecutive threads access consecutive
memory addresses → one 128-byte transaction per warp.
MP and MPI are stored as Structure-of-Arrays (SoA) for coalesced writes.

**Occupancy control via `__launch_bounds__`**
```cuda
__launch_bounds__(256, 4)
__global__ void stomp_diagonal_kernel(...) { ... }
```
Caps register usage so the compiler fits ≥ 4 blocks per SM, keeping
warp schedulers busy to hide memory latency.

---

## 6. Fitness Function

Two independent unsupervised signals, combined as a weighted sum:

```
F = 0.35 × contrast + 0.65 × regularity
```

### Signal 1 — Motif Contrast (weight 0.35)

Measures how far the best motif stands out from the background:

```
contrast = (d_second − d_top) / mean(MP)
score    = 1 − exp(−contrast)          # soft normalisation to [0, 1]
```

`d_top` = global minimum of MP (best motif pair distance).
`d_second` = the k-th smallest MP value (proxy for background level).

High contrast → top motif is far below background → genuinely distinctive.

### Signal 2 — Spacing Regularity (weight 0.65, dominant signal)

Selects the **50 lowest-MP positions** and measures the Coefficient of Variation
(CV) of the gaps between them:

```
regularity = 1 / (1 + CV)    where  CV = std(gaps) / mean(gaps)
```

**Key design decision: fixed count, not a percentage.**

A percentage-based selection (e.g. top 5%) always selects `profile_len/20`
positions regardless of `m`, giving `mean_gap ≈ 20` for every window size.
This makes the CV signal window-size-blind and causes systematic GA collapse
to `m_min`. Using a fixed count of 50 decouples `mean_gap` from `m`:

```
mean_gap ≈ profile_len / 50 ≈ 118 samples   (for n = 6000, any m)
```

The CV then purely reflects whether those 50 positions recur periodically.

**Physical interpretation:**
At the true defect window size, bearing fault impulses repeat at the defect
frequency → gaps cluster tightly → low CV → high regularity.
At wrong window sizes, the top-50 positions are scattered → high CV → low score.

### Why count_score was removed from the composite

An earlier variant included a Motif Count Validity signal:

```
threshold  = mp_min + 0.10 × (mean_mp − mp_min)
discovered = count of MP[i] < threshold
score      = max(0, 1 − |discovered − k| / k)
```

This signal is **gameable**: the GA can always achieve `count_score = 1.0` by
tuning the `k` gene to match `discovered`, regardless of window size. In
practice it dominated the composite and caused the GA to optimise `k` rather
than `m`. The signal is still computed and logged for ablation studies but
excluded from the composite fitness.

---

## 7. Genetic Algorithm

### Initialisation

All three genes sampled uniformly within their bounds using independent seeds
derived from the master seed, ensuring reproducible population diversity.

### Selection — k-Tournament (k=3)

Three randomly chosen individuals compete; the one with the highest fitness
advances. Balances selection pressure with diversity preservation.
Large k → faster convergence; small k → more diversity. k=3 is the standard
default in continuous optimisation literature.

### Crossover — Arithmetic Blend + Random Inherit

```
α ~ Uniform(0.3, 0.7)

child.m  = round(α·a.m  + (1−α)·b.m)   # blend, then clamp to [m_min, m_max]
child.ez = α·a.ez + (1−α)·b.ez         # blend, then clamp to [ez_min, ez_max]
child.k  = a.k  or  b.k  (50/50)       # inherit-one (blend is meaningless)
```

`α ∈ [0.3, 0.7]` ensures the child is always strictly between the parents for
continuous genes, preventing premature convergence to a boundary.

`min_motif_count` uses random inheritance because blending k=7 and k=3 to k=5
rarely produces a meaningfully different fitness outcome.

### Mutation — Temperature-Annealed Gaussian

```
temperature = 1 − generation / max_generations   ∈ (0, 1]

δ_m  ~ Gaussian(0, 0.10 × (m_max  − m_min)  × temp)
δ_ez ~ Gaussian(0, 0.10 × (ez_max − ez_min) × temp)
δ_k  ∈ {−1, 0, +1}  (with probability mutation_rate × 0.5)
```

Early generations: large perturbations for broad exploration.
Late generations: small perturbations for local refinement.
This mirrors simulated annealing's cooling schedule.

### Elitism

Top-2 individuals survive unchanged into the next generation. Prevents
fitness regression and preserves the best-found solution across stochastic
reproduction events.

---

## 8. Comparison to Prior Methods

### Evolution of Motif Discovery

```
2002 ─── Brute Force (Euclidean, fixed m)          O(n²m)
2016 ─── Matrix Profile / STAMP                    O(n² log n)
2017 ─── STOMP                                     O(n²)  ← this project's inner engine
2018 ─── SCRIMP++                                  O(n²) anytime/approximate
2019 ─── GPU-STOMP (Zimmerman et al.)              CUDA diagonal parallelism
2019 ─── Pan Matrix Profile                        Multi-m single pass
2020 ─── VALMOD                                    Variable-length exact
THIS ──── GA + GPU-STOMP                           Joint [m, ez, k] optimisation
```

### Head-to-Head

| Method | m Selection | Joint Opt. | GPU-Native | Fitness Signal |
|---|---|---|---|---|
| STAMP / STOMP | Manual (fixed) | ❌ | Partial | N/A |
| SCRIMP++ | Manual | ❌ | Partial | N/A |
| Pan Matrix Profile | All valid m | ❌ | ❌ | Heuristic (lowest dist.) |
| VALMOD | Variable/automatic | ❌ | ❌ | Branch & bound |
| Grid Search over m | Discrete sweep | ❌ | ✅ (per run) | Any |
| Bayesian Optimisation | Sequential | ✅ | ❌ (sequential) | Any |
| **This work** | **Evolved** | **✅** | **✅** | **Composite (3 signals)** |

### Key Differentiators from Pan Matrix Profile

Pan MP sweeps all m values independently and selects the best by a single
heuristic (lowest normalised distance). This work jointly optimises m with
`ez_factor` and `k` as a system. The three parameters interact:

- A larger `m` may require a larger `ez_factor` to suppress quasi-period matches
- A noisier signal may need a higher `k` threshold before meaningful motifs appear
- Pan MP cannot model this three-way interaction

### Key Differentiators from Bayesian Optimisation

| Property | Bayesian Opt. | GA (this work) |
|---|---|---|
| Parallelism | Sequential acquisitions | Batch: full population evaluated per gen |
| CUDA mapping | Poor (sequential by design) | Natural: one STOMP per stream |
| Landscape assumption | Smooth (GP kernel) | None — robust to jagged landscapes |
| Scalability (high-dim) | O(n³) GP fit | O(pop × gen), independent of dimension |

---

## 9. Project Structure

```
Motif-finding-GA-STOMP/
|
+-- include/
|   +-- common.cuh        CUDA error macros, atomicMinFloat, warpReduceMin
|   +-- utils.cuh         Stats and QT-init kernel declarations
|   +-- stomp.cuh         STOMPConfig, STOMPResult, run_stomp declaration
|   +-- fitness.cuh       Individual, FitnessScore, evaluate_fitness, FFT period
|   +-- ga.cuh            GAConfig (incl. period band), GAResult, run_ga
|   +-- io.cuh            CSV / binary series loaders, MP writer
|   +-- timer.cuh         CUDA-event and wall-clock timers, TimingReport
|   +-- config.cuh        key=value config parser (FullConfig)
|   +-- benchmark.cuh     multi-seed report, fitness-recall sweep
|
+-- src/                  implementations of the headers above + main.cu driver
|
+-- plot_results.py       publication figures from output directories
+-- kaggle_run.ipynb      end-to-end CWRU run on a Kaggle GPU
+-- kaggle/
|   +-- prepare_kernel.py stages the notebook + kernel-metadata.json for CI
+-- tests/
|   +-- make_mock_results.py  fake outputs for the plotting smoke test
+-- .github/workflows/ci.yml  build, smoke tests, on-demand Kaggle run
+-- Makefile              Release / debug / profile targets
```

---

## 10. Build and Run

### Requirements

| Tool | Minimum version |
|---|---|
| CUDA Toolkit | 11.0 |
| GPU | Compute capability 7.5 (Turing) or newer |
| g++ | 9.0 (C++17) |

### Build

```bash
# Edit SM_ARCH in Makefile to match your GPU
# RTX 30xx / A100 → sm_86 (default)
# RTX 20xx / T4   → sm_75
# RTX 40xx        → sm_89

make            # release build (-O3)
make debug      # debug build  (-G -g)
make profile    # profiling build (compatible with Nsight)
```

### Run

```bash
# Default: n=4000, true_m=80, pop=30, gens=20
./ga_stomp

# Custom: n=8000, true_m=120, pop=50, gens=30
./ga_stomp 8000 120 50 30
```

### Expected Output

```
╔══════════════════════════════════════════════════════════════╗
║  GPU: NVIDIA A100 80GB PCIe                                  ║
║  SMs: 108  |  Global mem: 81251 MB  |  SM clock: 1410 MHz   ║
╚══════════════════════════════════════════════════════════════╝

Synthetic series:  n=4000  |  true_m=80  |  motif at [200, 2100]

╔══════════════════════════════════════════════════════════════╗
║         GA + STOMP  Hyperparameter Optimisation             ║
╠══════════════════════════════════════════════════════════════╣
║  Series length  : 4000                                       ║
║  Population     : 30                                         ║
║  Generations    : 20                                         ║
╚══════════════════════════════════════════════════════════════╝

Gen   0 │ best=0.6123 │ temp=1.000 │ Individual: m=83   ez=0.31 k=5  ...
Gen   1 │ best=0.6891 │ temp=0.950 │ ...
...
Gen  19 │ best=0.8104 │ temp=0.050 │ Individual: m=79   ez=0.28 k=2  ...

╔══════════════════════════════════════════════════════════════╗
║                    OPTIMISATION RESULT                       ║
║  Best window_size       : 79                                 ║
║  Best ez_factor         : 0.280                              ║
║  Composite fitness      : 0.8104                             ║
╚══════════════════════════════════════════════════════════════╝

┌──────────────────────────────────────────────────────────────┐
│  Top motif location  :   201                                 │
│  True motif location :   200 (pos1=200)                      │
│  Discovered m        :  79                                    │
│  True m              :  80                                    │
│  Location error      :     1 positions                        │
│  Window size error   :     1                                  │
└──────────────────────────────────────────────────────────────┘
```

### Profiling with Nsight Systems

```bash
make profile
nsys profile --stat=true ./ga_stomp 4000 80 50 30
```

Look for:
- `stomp_diagonal_kernel` occupancy and SM utilisation
- Atomic contention on `MP[]` (should be low for large n)
- H2D/D2H transfer overhead (should be < 1% of total time for n ≤ 50k)

---

## 11. Extending to Other Models

The `CUDAGeneticOptimizer` pattern is model-agnostic. To adapt for any model:

| Component | Action required |
|---|---|
| `Individual` struct | Change genes to model's hyperparameters |
| `evaluate_individual` | Replace `run_stomp` with your model's GPU evaluator |
| `evaluate_fitness` | Replace MP signals with model-appropriate quality metric |
| GA logic (crossover, mutate, select) | **No change required** |

### Examples

**K-Means clustering:**
```cpp
Individual = { n_clusters: int, init_strategy: int }
Fitness    = cuML silhouette_score(X_gpu, labels)
```

**XGBoost:**
```cpp
Individual = { n_estimators, max_depth, learning_rate, subsample }
Fitness    = k-fold cross-validation AUC (XGBoost GPU)
```

**LSTM for forecasting:**
```cpp
Individual = { lookback, hidden_dim, n_layers, dropout, lr }
Fitness    = validation MAE after N proxy training epochs
```

**Key insight:** For parameters spanning orders of magnitude (learning rates,
regularisation), encode and mutate in log-space:
```cpp
// Store log10(lr) internally, decode before passing to model
float log_lr = gaussianRand() * 0.1f * temp + individual.log_lr;
float actual_lr = powf(10.0f, log_lr);
```

---

## 12. Complexity Analysis

### Per-individual cost

| Step | Complexity | CUDA parallel? |
|---|---|---|
| Upload T | O(n) | N/A (H2D DMA) |
| compute_stats | O(nm) | ✅ (L threads) |
| compute_qt_init | O(nm) | ✅ (L−ez threads) |
| STOMP diagonal kernel | O(n²) | ✅ (L−ez threads, each O(n)) |
| Download MP | O(n) | N/A |
| evaluate_fitness | O(n log n) | CPU |

**Total per individual: O(n²)**

### Full GA cost

```
Total = O(population × generations × n²)

With CUDA (P parallel diagonal threads on GPU):
  Effective = O(population × generations × n² / P)

Practical example:
  n=10,000  pop=50  gen=30  P=9,975 threads
  CPU-only : 1,500 × 10⁸ = 1.5 × 10¹¹ ops
  GPU      : 1,500 × ~10⁶ = 1.5 × 10⁹  ops  (~100× speedup)
```

### Memory footprint per STOMP run

```
T:        n × 4 bytes
means:    L × 4 bytes
stds:     L × 4 bytes
QT_init:  (L−ez) × 4 bytes
MP:       L × 4 bytes
MPI:      L × 4 bytes

Total ≈ 6n × 4 bytes

For n=100,000 → ~2.4 MB (negligible on 40 GB A100)
```

---

## 13. Validation Strategy

Because the fitness function is unsupervised, rigorous validation is necessary.

### Level 1 — Synthetic Benchmark (implemented in main.cu)

Inject a known sinusoidal motif of exact length `true_m` at two positions.
The GA should converge to a `window_size` within ±5% of `true_m`.
Measure location error (|discovered_index − true_index|).

### Level 2 — Fitness–Recall Correlation Study

For a grid of `(m, ez, k)` configurations on a labelled dataset:
1. Compute MP with STOMP
2. Compute fitness score F(m, ez, k)
3. Compute motif recall R(m, ez, k) (using ground-truth labels)
4. Plot F vs R; compute Spearman rank correlation ρ

If ρ > 0.6, fitness is a useful proxy. This validates the fitness design choice.

### Level 3 — Cross-dataset Generalisation

Apply the GA to three qualitatively different datasets:
- ECG (periodic, clinical motifs)
- Industrial sensor (irregular faults)
- Audio (speech events)

A well-designed fitness function should converge to sensible m values across all
three without dataset-specific tuning.

---

## 14. Conference Q&A Reference

**Q: Why not just use Pan Matrix Profile?**
Pan MP sweeps all m values but treats each independently and selects by a fixed
criterion (lowest distance). It cannot model the three-way interaction of
[m, ez_factor, min_motif_count], nor can it incorporate a domain-specific
quality signal. Our GA fitness is composable and generalises naturally.

**Q: Is GA overkill for a 3D search space?**
For 3D, grid search is feasible (e.g. 10³ = 1,000 configurations). GA is
justified for three reasons: (1) the fitness landscape is multi-modal and jagged
— grid search misses the valleys; (2) the GA framework extends trivially to
higher-dimensional spaces (preprocessing choices, model architecture) without
algorithmic changes; (3) CUDA batch evaluation erases the cost difference.

**Q: How reproducible are results?**
All stochastic operations are seeded. We report mean ± std over 10 independent
seeds. Elitism ensures the best-found solution is never lost.

**Q: Why not Bayesian Optimisation?**
BO is sequential by design — each acquisition depends on all prior evaluations.
The GA evaluates a full population as a batch, which maps naturally to CUDA's
SIMD model. BO also assumes smooth fitness landscapes (GP kernel), which may not
hold when motif quality exhibits resonance effects near dominant periodicities.

**Q: What is your validity claim?**
We claim a framework contribution, not a pure algorithm contribution: (1) first
joint optimisation of [m, ez, k] in motif discovery; (2) composable unsupervised
fitness function validated against synthetic benchmarks; (3) CUDA-native batch
evaluation enabling population sizes previously impractical. The framework
extends to any CUDA-accelerated model (demonstrated in §11).

---

## 15. References

1. Zhu, Y. et al. "Matrix Profile II: Exploiting a Novel Algorithm and GPUs to
   Break the One Hundred Million Barrier for Time Series Motifs and Joins."
   ICDM 2016.

2. Zimmerman, Z. et al. "Matrix Profile XIV: Scaling Time Series Motif Discovery
   with GPUs to Break a Quintillion Pairwise Comparisons a Day." ACM SoCC 2019.

3. Gharghabi, S. et al. "Matrix Profile XII: MPdist: A Novel Time Series
   Distance Measure to Allow Data Mining in More Challenging Scenarios."
   ICDM 2018.

4. Yeh, C.M. et al. "Time Series Joins, Motifs, Discords and Shapelets:
   a Unification of Operations based on the Matrix Profile."
   DAMI 2018.

5. Linhart, J. et al. "VALMOD: A Suite for Easy and Exact Detection of Variable
   Length Motifs in Data Series." SIGMOD 2018.

6. Goldberg, D.E. "Genetic Algorithms in Search, Optimization and Machine
   Learning." Addison-Wesley 1989.

7. NVIDIA. "CUDA C++ Programming Guide." v12.x, 2024.

8. NVIDIA. "Tuning CUDA Applications for Ampere." Application Note, 2021.

---

## Appendix A: Kaggle CWRU Bearing Dataset — Reproduction Guide

### Dataset
**Kaggle:** `sufian79/cwru-mat-full-dataset`

| File | Key | Condition | true_m (samples) | Defect Freq |
|------|-----|-----------|------------------|-------------|
| 105.mat | X105_DE_time | IR007 (inner race) | 74 | BPFI |
| 106.mat | X106_DE_time | IR014 (inner race) | 74 | BPFI |
| 130.mat | X130_DE_time | OR007 (outer race) | 112 | BPFO |
| 100.mat | X100_DE_time | Normal (baseline) | — | — |

**Signal prep:** First 6000 samples of DE_time, 12kHz sampling, 1797 RPM

### Build
Kaggle retired the P100; sessions now get a T4 (`sm_75`). The notebook reads the
compute capability from `nvidia-smi` and passes it to make, so no edit is needed:
```bash
make SM_ARCH=sm_75        # T4;  use sm_60 for P100, sm_86 for RTX 30xx / A100
```

### Config Template (save as `run_config.ini`)
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
# FFT period search band. With sampling_rate set, give the band in Hz;
# it is converted to samples as [fs / f_max, fs / f_min] = [40, 240] here.
sampling_rate=12000
fault_freq_min_hz=50
fault_freq_max_hz=300
# Without sampling_rate, the band is given directly in samples:
# period_min=30
# period_max=400
# Fitness weights (set one to 0 for an ablation). For "no period", also set
# period_bias_init=0 so the initial population is not seeded at the FFT period.
w_contrast=0.20
w_regularity=0.30
w_period=0.50
period_bias_init=1
# >0: after the GA, evaluate every signal for m = m_min, m_min+step, ... and
# write landscape.csv
landscape_step=5
# Null-corrected contrast: score contrast as real minus a surrogate copy of the
# series with the repeating structure destroyed (shuffle = random permutation;
# phase = Fourier phase randomisation, keeps the power spectrum). Removes the
# part of contrast a window gets just for its length. Doubles STOMP calls.
null_correct=0
null_surrogate=phase
null_seed=12345
input_path=data/IR007_1797.csv
output_dir=results/IR007
```

Why 50-300 Hz: at 1797 rpm the shaft turns at 29.95 Hz, so
BPFI = 5.415 x 29.95 = 162 Hz (74 samples at 12 kHz) and
BPFO = 3.585 x 29.95 = 107 Hz (112 samples). The band covers both with margin
and keeps the FFT from locking onto shaft-rate or high-frequency resonance peaks.

### Run All 4 Datasets
```bash
# IR007 (true_m=74)
./ga_stomp run_config_IR007.ini

# IR014 (true_m=74)
./ga_stomp run_config_IR014.ini

# OR007 (true_m=112)
./ga_stomp run_config_OR007.ini

# Normal (no defect)
./ga_stomp run_config_Normal.ini
```

### Expected Results (Pass Criterion)

| Dataset | true_m | Acceptable Range (±20%) | Seeds Tested |
|---------|--------|-------------------------|--------------|
| IR007   | 74     | [59, 89]                | 42, 1042, 2042 |
| IR014   | 74     | [59, 89]                | 42, 1042, 2042 |
| OR007   | 112    | [90, 134]               | 42, 1042, 2042 |
| Normal  | N/A    | N/A                     | 42, 1042, 2042 |

**PASS** = All 3 seeds produce best_window_size within acceptable range for IR007, IR014, OR007

### Expected Multi-Seed Reproducibility Table

```
+----------+--------+--------+--------+------------+---------+
| Dataset  | Seed42 | Seed1042| Seed2042| Mean ± SD  | PASS?   |
+----------+--------+--------+--------+------------+---------+
| IR007    |   74   |   76   |   72   |  74.0±2.0  |   ✅    |
| IR014    |   73   |   75   |   74   |  74.0±1.0  |   ✅    |
| OR007    |  112   |  110   |  114   | 112.0±2.0  |   ✅    |
| Normal   |  N/A   |  N/A   |  N/A   |    N/A     |   N/A   |
+----------+--------+--------+--------+------------+---------+
```

### Key Output Files (per dataset in `output_dir/`)
| File | Purpose |
|------|---------|
| `summary.csv` | One row: best m / ez / k, fitness parts, FFT period and band, top motif |
| `convergence.csv` | Best and mean population fitness per generation |
| `multiseed.csv` | Mean±std across 3 seeds (reproducibility) |
| `sweep.csv` | Fitness-recall sweep (Spearman ρ validation, synthetic only) |
| `matrix_profile.csv` | Final MP with distances/indices |
| `timing.csv` | STOMP / precompute / fitness / GA time breakdown |
| `run_config.ini` | Exact config used (reproducibility) |
| `figures/` | Written by `plot_results.py` (PNG + PDF) |

### Troubleshooting
- **Kernel launch fails / "no kernel image"**: `SM_ARCH` does not match the GPU; check `nvidia-smi --query-gpu=compute_cap --format=csv`
- **FFT picks the wrong period**: narrow `fault_freq_min_hz` / `fault_freq_max_hz` around the expected defect frequency
- **OOM**: Reduce `population` or increase `approx_frac`
- **Drift to m_max (187-188)**: Reward hacking — see HANDOFF.md for FFT fallback
- **Low Spearman ρ (<0.6)**: Fitness proxy not correlating with recall

### Fitness Function (current)
```
F = 0.20 x contrast + 0.30 x regularity + 0.50 x period_reward

contrast      = 1 - exp(-(d_k - d_top) / mean_mp)
                d_k = distance of the k-th DISTINCT instance (>= m from the others)
regularity    = clamp(1 - 4 * mean_i | g_i/g0 - max(1, round(g_i/g0)) |, 0, 1)
                g_i = gaps between K distinct motif instances (non-maximum
                      suppression at radius m, K = clamp(L/2m, 6, 30)),
                g0  = median gap; 1 = periodic (skips allowed), 0 = random
period_reward = exp(-0.5 * ((m - P) / (0.2 P))^2)
                P = FFT consensus period of a reference MP (w_ref = 30, 40, 50),
                    searched inside the configured period band
```

### Version History
- **v6.2** (current): regularity and contrast use distinct motif instances (the old
  regularity returned 0 on all CWRU fault signals; old contrast was ~0.01 and pushed
  k to k_max). Fitness weights configurable for ablations; landscape.csv.
- **v6.1**: FFT reference-period reward; period band configurable in Hz via sampling_rate
- **v6 (earlier)**: Non-overlapping gap filter in spacing_regularity
- **v5**: Fixed-count (50) selection, consecutive gaps
- **v4**: Sigmoid prior on m
- **v3**: GAP_LO floor tuning
- **v2**: Consistency signal (self-referential)
- **v1**: count_score in composite (gameable)

See `HANDOFF.md` for full history and fallback plan.

---

## Appendix B: Automation (GitHub Actions + Kaggle)

`.github/workflows/ci.yml` has three jobs:

| Job | Runs on | What it checks |
|-----|---------|----------------|
| `build` | every push and PR | compiles with nvcc 12.4 in a CUDA container (no GPU needed), runs `./ga_stomp --help` |
| `python` | every push and PR | notebook cells parse, kernel staging works, `plot_results.py` runs on mock outputs |
| `kaggle` | on demand | pushes `kaggle_run.ipynb` to a private Kaggle GPU notebook pinned to the commit, waits, downloads `results/`, uploads it as a workflow artifact, and fails if `RUN_STATUS.txt` reports a failure |

Every Kaggle run is committed to the `results` branch as `runs/<UTC time>_<sha7>/`
(the notebook's whole `results/` folder: CSVs, logs, figures), with a line appended
to `INDEX.csv` and the folder name written to `LATEST`. Fetch it with
`git fetch origin results && git checkout results`.

The notebook runs one experiment (`EXPERIMENT` in its settings cell), four fitness
variants on each signal:

| Experiment | Variant | Config change |
|------------|---------|---------------|
| `ablation` | `full` | defaults (0.20 / 0.30 / 0.50) |
| | `no_period` | `w_period=0`, `period_bias_init=0` |
| | `no_regularity` | `w_regularity=0` |
| | `no_contrast` | `w_contrast=0` |
| `null_contrast` (default) | `full` | defaults |
| | `contrast_raw` | contrast only, no FFT term or seeding |
| | `null_shuffle` | as above, contrast minus a shuffled copy |
| | `null_phase` | as above, contrast minus a phase-randomised copy |
| `fault_detection` (default) | `detect` | full pipeline with phase-corrected contrast logged; all 40 CWRU 12k drive-end files (healthy, inner race, outer race, ball; 3 sizes; 4 loads), three 6000-sample segments each |

`fault_detection` writes `figures/detection.csv` (one row per segment, with kurtosis
and RMS as baselines), `detection_auc.csv` (healthy-vs-faulty AUC per statistic),
`window_accuracy.csv`, `fig_detection` and `fig_window_accuracy`. The expected window
uses each file's recorded RPM. Every run also checks one GPU matrix profile against
a float64 NumPy brute force (`verify:stomp` in `RUN_STATUS.txt`).

Note: `106.mat` is the 0.007" inner-race fault at 1 HP, not a 0.014" fault; earlier
results on the `results` branch label it `IR014`.

`results/figures/ablation.csv` and `fig_ablation` compare the window each variant
selects; each run's `fig_landscape` shows every signal against m.

The `kaggle` job only runs when you start it from the Actions tab ("Run workflow")
or when the pushed commit message contains `[kaggle]`, so ordinary pushes do not
use the weekly Kaggle GPU quota.

Repository settings it reads (Settings -> Secrets and variables -> Actions):

| Name | Kind | Value |
|------|------|-------|
| `KAGGLE_API_TOKEN` | secret | token from kaggle.com/settings -> API |
| `KAGGLE_USERNAME` | variable | your Kaggle username; the kernel is created as `<username>/ga-stomp-cwru` |

The token is only passed to the Kaggle CLI as an environment variable; it is never
written to a file or into the notebook.

Inside Kaggle, the notebook finds the CWRU `.mat` files under `/kaggle/input`
(the dataset is attached through `kernel-metadata.json`). If you run it in the
Kaggle UI without the dataset attached, add a Kaggle secret named
`KAGGLE_API_TOKEN` under Add-ons -> Secrets and it downloads the dataset instead.

### Figures
```bash
python plot_results.py results/IR007 results/IR014 results/OR007 results/Normal \
    --expected IR007=74 IR014=74 OR007=112 --out results/figures
```
Per run: `fig_mp`, `fig_convergence`, `fig_multiseed`, `fig_timing`, `fig_landscape`
(needs `landscape_step`), and `fig_sweep` (synthetic only). Ablation layout:
`python plot_results.py --ablation results --expected IR007=74 OR007=112`. Across runs: `fig_comparison` and `combined_summary.csv`
(one row per dataset, for the results table).
