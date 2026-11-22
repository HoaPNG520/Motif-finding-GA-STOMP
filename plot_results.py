#!/usr/bin/env python3
"""
plot_results.py -- publication figures from GA-STOMP output directories.

Usage
    python plot_results.py results/IR007
    python plot_results.py results/IR007 results/OR007 results/Normal \
        --expected IR007=74 IR014=74 OR007=112 --out figures

For each result directory (the output_dir of one ga_stomp run) it writes
    <dir>/figures/fig_mp.{png,pdf}            series + matrix profile, top motif marked
    <dir>/figures/fig_convergence.{png,pdf}   best / mean fitness per generation
    <dir>/figures/fig_multiseed.{png,pdf}     best window size per seed vs FFT period
    <dir>/figures/fig_sweep.{png,pdf}         fitness and recall vs m (synthetic only)
    <dir>/figures/fig_timing.{png,pdf}        where the wall time went
    <dir>/figures/fig_landscape.{png,pdf}     every fitness signal vs m (needs landscape.csv)

With two or more directories it also writes, into --out (default: figures/)
    fig_comparison.{png,pdf}   GA window vs FFT period (vs expected) per dataset
    combined_summary.csv       one row per dataset, for the paper's results table

Ablation mode
    python plot_results.py --ablation results --expected IR007=74 OR007=112
reads results/<variant>/<dataset>/summary.csv and writes results/figures/
ablation.csv and fig_ablation.{png,pdf} (window chosen per dataset and variant).

Every input file is optional; a figure whose CSV is missing is skipped.
"""

import argparse
import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Categorical slots in fixed order (validated: CVD + normal-vision separation pass).
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "lines.linewidth": 1.5,
    "legend.frameon": False,
    "legend.fontsize": 8,
})


# ---------------------------------------------------------------------------
#  CSV helpers
# ---------------------------------------------------------------------------
def read_csv(path):
    """Read a CSV that may start with '# comment' lines. Returns (rows, comments)."""
    if not os.path.isfile(path):
        return None, []
    comments, lines = [], []
    with open(path, newline="") as f:
        for line in f:
            if line.startswith("#"):
                comments.append(line[1:].strip())
            elif line.strip():
                lines.append(line)
    if not lines:
        return [], comments
    return list(csv.DictReader(lines)), comments


def col(rows, name, cast=float):
    out = []
    for r in rows:
        v = r.get(name, "")
        try:
            out.append(cast(v))
        except (TypeError, ValueError):
            out.append(np.nan)
    return np.array(out)


def save(fig, out_dir, stem):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{stem}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.join(out_dir, stem)}.png/.pdf")


def end_label(ax, x, y, text, color):
    """Direct label at the right end of a line (text stays in ink colour)."""
    ax.annotate(text, (x, y), xytext=(4, 0), textcoords="offset points",
                va="center", fontsize=8, color=INK2)
    ax.plot([x], [y], "o", ms=4, color=color, mec="white", mew=1)


def load_summary(d):
    rows, _ = read_csv(os.path.join(d, "summary.csv"))
    return rows[0] if rows else None


def ds_name(d):
    return os.path.basename(os.path.normpath(d))


# ---------------------------------------------------------------------------
#  Per-run figures
# ---------------------------------------------------------------------------
def fig_mp(d, out, summary):
    series, _ = read_csv(os.path.join(d, "series.csv"))
    mp, comments = read_csv(os.path.join(d, "matrix_profile.csv"))
    if not series or not mp:
        return
    t = col(series, "value")
    dist = col(mp, "mp_distance")
    nn = col(mp, "mp_index", int) if "mp_index" in mp[0] else None
    finite = np.isfinite(dist)
    if not finite.any():
        return

    m = int(summary["best_m"]) if summary else None
    if m is None:
        for c in comments:  # fall back to the "window_size=" header if present
            if "window" in c and "=" in c:
                try:
                    m = int(c.split("=")[-1].split()[0])
                except ValueError:
                    pass
    top = int(np.nanargmin(np.where(finite, dist, np.inf)))
    partner = int(nn[top]) if nn is not None and nn[top] >= 0 else None

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 3.6), sharex=True,
                                 gridspec_kw={"height_ratios": [1, 1], "hspace": 0.12})
    a1.plot(np.arange(len(t)), t, color=INK2, lw=0.6)
    if m:
        a1.axvspan(top, top + m, color=BLUE, alpha=0.25, lw=0)
        a1.annotate("top motif", (top + m / 2, a1.get_ylim()[1]), ha="center",
                    va="bottom", fontsize=8, color=INK2, annotation_clip=False)
        if partner is not None:
            a1.axvspan(partner, partner + m, color=ORANGE, alpha=0.25, lw=0)
            a1.annotate("nearest match", (partner + m / 2, a1.get_ylim()[1]),
                        ha="center", va="bottom", fontsize=8, color=INK2,
                        annotation_clip=False)
    a1.set_ylabel("value")
    a1.set_title(f"{ds_name(d)}  (m = {m})" if m else ds_name(d), loc="left", pad=14)

    x = np.arange(len(dist))
    a2.plot(x[finite], dist[finite], color=BLUE)
    a2.plot([top], [dist[top]], "o", ms=6, color=BLUE, mec="white", mew=1.5)
    a2.annotate(f"min {dist[top]:.2f} @ {top}", (top, dist[top]), xytext=(6, 6),
                textcoords="offset points", fontsize=8, color=INK2)
    a2.set_ylabel("MP distance")
    a2.set_xlabel("sample index")
    save(fig, out, "fig_mp")


def fig_convergence(d, out):
    rows, _ = read_csv(os.path.join(d, "convergence.csv"))
    if not rows:
        return
    g = col(rows, "generation", int)
    best, mean = col(rows, "best_fitness"), col(rows, "mean_fitness")
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    ax.plot(g, best, color=BLUE, label="best")
    ax.plot(g, mean, color=ORANGE, label="population mean")
    end_label(ax, g[-1], best[-1], f"best {best[-1]:.3f}", BLUE)
    end_label(ax, g[-1], mean[-1], f"mean {mean[-1]:.3f}", ORANGE)
    ax.set_xlabel("generation")
    ax.set_ylabel("composite fitness")
    ax.set_title(f"GA convergence -- {ds_name(d)}", loc="left")
    ax.legend(loc="lower right")
    ax.margins(x=0.15)
    save(fig, out, "fig_convergence")


def fig_multiseed(d, out, summary, expected):
    rows, _ = read_csv(os.path.join(d, "multiseed.csv"))
    if not rows:
        return
    rows = [r for r in rows if r.get("seed") not in ("MEAN", "STD")]
    if not rows:
        return
    seeds = [r["seed"] for r in rows]
    m = col(rows, "best_m")
    x = np.arange(len(m))
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    mu, sd = float(np.nanmean(m)), float(np.nanstd(m))
    ax.axhspan(mu - sd, mu + sd, color=BLUE, alpha=0.12, lw=0)
    ax.axhline(mu, color=BLUE, lw=1)
    ax.plot(x, m, "o", ms=8, color=BLUE, mec="white", mew=1.5)
    ax.set_xticks(x, seeds)
    ax.margins(x=0.35)
    xl = ax.get_xlim()
    ax.annotate(f"mean {mu:.1f} +/- {sd:.1f}", (xl[1], mu), xytext=(-3, 3),
                textcoords="offset points", ha="right", va="bottom",
                fontsize=8, color=INK2)
    refs = []
    if summary and int(summary.get("detected_period", -1)) > 0:
        refs.append(("FFT period", int(summary["detected_period"]), ORANGE))
    if expected is not None:
        refs.append(("expected", expected, AQUA))
    for label, val, c in refs:
        ax.axhline(val, color=c, lw=1.5, ls="--")
        ax.annotate(f"{label} {val}", (xl[0], val), xytext=(3, 3),
                    textcoords="offset points", ha="left", va="bottom",
                    fontsize=8, color=INK2)
    ax.set_xlim(xl)
    ax.set_xlabel("seed")
    ax.set_ylabel("best window size m")
    ax.set_title(f"Reproducibility across seeds -- {ds_name(d)}", loc="left")
    save(fig, out, "fig_multiseed")


def fig_sweep(d, out, summary):
    rows, comments = read_csv(os.path.join(d, "sweep.csv"))
    if not rows:
        return
    m = col(rows, "m", int)
    fit, rec = col(rows, "fitness"), col(rows, "recall")
    rho = next((c.split("=")[-1].strip() for c in comments if "Spearman" in c), "n/a")
    true_m = int(summary["true_m"]) if summary and int(summary.get("true_m", 0)) > 0 else None

    # Two panels, one y-axis each (never a dual axis).
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(4.6, 3.8), sharex=True,
                                 gridspec_kw={"hspace": 0.15})
    a1.plot(m, fit, "-o", ms=4, color=BLUE)
    a1.set_ylabel("composite fitness")
    a1.set_title(f"Fitness vs recall sweep  (Spearman rho = {rho})", loc="left")
    a2.bar(m, rec, width=max(1, (m.max() - m.min()) / max(len(m), 1) * 0.7),
           color=AQUA, edgecolor="white", linewidth=1)
    a2.set_ylabel("recall")
    a2.set_ylim(0, 1.25)
    a2.set_xlabel("window size m")
    if true_m:
        for a in (a1, a2):
            a.axvline(true_m, color=ORANGE, ls="--", lw=1.5)
        a2.annotate(f"true m = {true_m}", (true_m, 1.12), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8, color=INK2)
    save(fig, out, "fig_sweep")


def fig_timing(d, out):
    rows, _ = read_csv(os.path.join(d, "timing.csv"))
    if not rows:
        return
    vals = {r["metric"]: r["value_ms"] for r in rows}
    parts = [("STOMP kernel", "stomp_total"), ("precompute", "precompute_total"),
             ("fitness (CPU)", "fitness_total"), ("GA operators", "ga_overhead")]
    labels, ms = [], []
    for lab, key in parts:
        try:
            ms.append(float(vals[key]))
            labels.append(lab)
        except (KeyError, ValueError):
            pass
    if not ms:
        return
    total = sum(ms)
    fig, ax = plt.subplots(figsize=(4.6, 2.2))
    y = np.arange(len(ms))[::-1]
    ax.barh(y, ms, color=BLUE, height=0.6, edgecolor="white", linewidth=1)
    for yi, v in zip(y, ms):
        ax.annotate(f"{v:,.0f} ms  ({100 * v / total:.0f}%)", (v, yi), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y, labels)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("time (ms)")
    calls = vals.get("n_stomp_calls", "?")
    ax.set_title(f"Time breakdown -- {ds_name(d)}  ({calls} STOMP calls)", loc="left")
    ax.margins(x=0.35)
    save(fig, out, "fig_timing")


def fig_landscape(d, out, expected):
    rows, comments = read_csv(os.path.join(d, "landscape.csv"))
    if not rows:
        return
    m = col(rows, "m", int)
    period = None
    for c in comments:
        if c.startswith("detected_period"):
            try:
                period = int(c.split("=")[-1])
            except ValueError:
                pass
    panels = [("composite", "composite"), ("contrast", "contrast"),
              ("regularity", "regularity"), ("period_reward", "period reward")]
    # Small multiples, one y-axis each; every signal is in [0, 1].
    fig, axes = plt.subplots(len(panels), 1, figsize=(5.2, 5.6), sharex=True,
                             gridspec_kw={"hspace": 0.35})
    for ax, (key, label) in zip(axes, panels):
        y = col(rows, key)
        ax.plot(m, y, color=BLUE)
        best = int(np.nanargmax(y))
        ax.plot([m[best]], [y[best]], "o", ms=5, color=BLUE, mec="white", mew=1)
        ax.annotate(f"max at m={m[best]}", (m[best], y[best]), xytext=(5, -2),
                    textcoords="offset points", va="top", fontsize=7, color=INK2)
        ax.set_title(label, loc="left", fontsize=8, pad=2)
        lo, hi = np.nanmin(y), np.nanmax(y)
        ax.set_ylim(min(0, lo), max(1.0, hi) * 1.08)
        if period and period > 0:
            ax.axvline(period, color=ORANGE, ls="--", lw=1.2)
        if expected:
            ax.axvline(expected, color=AQUA, ls=":", lw=1.5)
    top = axes[0]
    ytxt = top.get_ylim()[1]
    if period and period > 0:
        top.annotate(f"FFT {period}", (period, ytxt), xytext=(3, 2), textcoords="offset points",
                     fontsize=7, color=INK2, annotation_clip=False)
    if expected:
        top.annotate(f"expected {expected}", (expected, ytxt), xytext=(3, 10),
                     textcoords="offset points", fontsize=7, color=INK2, annotation_clip=False)
    axes[-1].set_xlabel("window size m")
    fig.suptitle(f"Fitness landscape -- {ds_name(d)}", x=0.02, ha="left", fontsize=10, y=0.99)
    save(fig, out, "fig_landscape")


# ---------------------------------------------------------------------------
#  Ablation: root/<variant>/<dataset>/summary.csv
# ---------------------------------------------------------------------------
VARIANT_ORDER = ["full", "no_period", "no_regularity", "no_contrast"]


def fig_ablation(root, expected_map):
    variants = [v for v in sorted(os.listdir(root))
                if os.path.isdir(os.path.join(root, v)) and v != "figures"]
    variants.sort(key=lambda v: (VARIANT_ORDER.index(v) if v in VARIANT_ORDER else 99, v))
    table = []
    for v in variants:
        for ds in sorted(os.listdir(os.path.join(root, v))):
            d = os.path.join(root, v, ds)
            s = load_summary(d) if os.path.isdir(d) else None
            if not s:
                continue
            ms_rows, _ = read_csv(os.path.join(d, "multiseed.csv"))
            stats = {r.get("seed"): r for r in ms_rows or []}
            exp = expected_map.get(ds)
            table.append({
                "dataset": ds, "variant": v, "best_m": int(s["best_m"]),
                "multiseed_mean_m": stats.get("MEAN", {}).get("best_m", ""),
                "multiseed_std_m": stats.get("STD", {}).get("best_m", ""),
                "detected_period": s["detected_period"],
                "expected_m": exp if exp is not None else "",
                "abs_error": abs(int(s["best_m"]) - exp) if exp is not None else "",
                "fitness": s["fitness"], "contrast": s["contrast"],
                "regularity": s["regularity"], "period_reward": s["period_reward"],
                "w_contrast": s.get("w_contrast", ""), "w_regularity": s.get("w_regularity", ""),
                "w_period": s.get("w_period", ""), "period_bias_init": s.get("period_bias_init", ""),
            })
    if not table:
        print(f"[ablation] no <variant>/<dataset>/summary.csv under {root}")
        return
    out = os.path.join(root, "figures")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "ablation.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)
    print(f"  wrote {path}")

    datasets = sorted({r["dataset"] for r in table},
                      key=lambda x: (expected_map.get(x) is None, x))
    palette = [BLUE, ORANGE, AQUA, YELLOW]
    variants = [v for v in variants if any(r["variant"] == v for r in table)][:4]
    k = len(variants)
    width = 0.8 / k
    x = np.arange(len(datasets))
    fig, ax = plt.subplots(figsize=(max(5.0, 1.8 * len(datasets) + 1.5), 3.2))
    for i, v in enumerate(variants):
        vals = []
        for ds in datasets:
            r = next((r for r in table if r["dataset"] == ds and r["variant"] == v), None)
            vals.append(r["best_m"] if r else np.nan)
        xs = x - 0.4 + width * (i + 0.5)
        ax.bar(xs, vals, width=width * 0.92, color=palette[i], label=v,
               edgecolor="white", linewidth=1)
        for xi, val in zip(xs, vals):
            if np.isfinite(val):
                ax.annotate(f"{val:.0f}", (xi, val), xytext=(0, 2), textcoords="offset points",
                            ha="center", fontsize=7, color=INK2)
    for xi, ds in zip(x, datasets):
        e = expected_map.get(ds)
        if e:
            ax.plot([xi - 0.42, xi + 0.42], [e, e], color=INK, lw=1.2, ls="--")
            ax.annotate(f"expected {e}", (xi + 0.42, e), xytext=(2, 0),
                        textcoords="offset points", va="center", fontsize=7, color=INK2)
    ax.set_xticks(x, datasets)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("selected window m (samples)")
    ax.set_title("Ablation: window chosen with each fitness term removed", loc="left")
    ax.legend(ncol=k, loc="upper left", bbox_to_anchor=(0, -0.12))
    ax.margins(x=0.08)
    save(fig, out, "fig_ablation")


# ---------------------------------------------------------------------------
#  Cross-dataset comparison
# ---------------------------------------------------------------------------
def fig_comparison(dirs, out, expected_map):
    table = []
    for d in dirs:
        s = load_summary(d)
        if not s:
            continue
        name = ds_name(d)
        ms_rows, _ = read_csv(os.path.join(d, "multiseed.csv"))
        mean_m = std_m = ""
        for r in ms_rows or []:
            if r.get("seed") == "MEAN":
                mean_m = r.get("best_m", "")
            if r.get("seed") == "STD":
                std_m = r.get("best_m", "")
        exp = expected_map.get(name)
        table.append({
            "dataset": name,
            "n": s["n"],
            "best_m": s["best_m"],
            "multiseed_mean_m": mean_m,
            "multiseed_std_m": std_m,
            "detected_period": s["detected_period"],
            "expected_m": exp if exp is not None else "",
            "abs_error": abs(int(s["best_m"]) - exp) if exp is not None else "",
            "best_ez": s["best_ez"],
            "best_k": s["best_k"],
            "fitness": s["fitness"],
            "wall_ms": s["wall_ms"],
        })
    if not table:
        return
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "combined_summary.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)
    print(f"  wrote {path}")

    names = [r["dataset"] for r in table]
    series = [("GA best m", [float(r["best_m"]) for r in table], BLUE),
              ("FFT period", [float(r["detected_period"]) for r in table], ORANGE)]
    if any(r["expected_m"] != "" for r in table):
        series.append(("expected", [float(r["expected_m"]) if r["expected_m"] != ""
                                    else np.nan for r in table], AQUA))
    k = len(series)
    width = 0.8 / k
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(4.5, 1.4 * len(names) + 1.5), 3.0))
    for i, (lab, vals, c) in enumerate(series):
        xs = x - 0.4 + width * (i + 0.5)
        ax.bar(xs, vals, width=width * 0.92, color=c, label=lab,
               edgecolor="white", linewidth=1)
        for xi, v in zip(xs, vals):
            if np.isfinite(v) and v > 0:
                ax.annotate(f"{v:.0f}", (xi, v), xytext=(0, 2), textcoords="offset points",
                            ha="center", fontsize=7, color=INK2)
    ax.set_xticks(x, names)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("window size (samples)")
    ax.set_title("Selected window vs FFT period per dataset", loc="left")
    ax.legend(ncol=k, loc="upper left", bbox_to_anchor=(0, -0.12))
    save(fig, out, "fig_comparison")


# ---------------------------------------------------------------------------
def parse_expected(items):
    out = {}
    for it in items or []:
        if "=" not in it:
            sys.exit(f"--expected entries look like NAME=74, got '{it}'")
        k, v = it.split("=", 1)
        out[k.strip()] = int(v)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="*", help="ga_stomp output directories")
    ap.add_argument("--ablation", metavar="ROOT",
                    help="ROOT/<variant>/<dataset>/ layout: write ablation.csv + fig_ablation")
    ap.add_argument("--expected", nargs="*", metavar="NAME=M",
                    help="known window size per dataset (dir basename), e.g. OR007=112")
    ap.add_argument("--out", default="figures",
                    help="where cross-dataset outputs go (default: figures/)")
    args = ap.parse_args()
    expected = parse_expected(args.expected)

    for d in args.dirs:
        if not os.path.isdir(d):
            print(f"[skip] {d}: not a directory")
            continue
        print(f"[{ds_name(d)}]")
        fig_out = os.path.join(d, "figures")
        s = load_summary(d)
        fig_mp(d, fig_out, s)
        fig_convergence(d, fig_out)
        fig_multiseed(d, fig_out, s, expected.get(ds_name(d)))
        fig_sweep(d, fig_out, s)
        fig_timing(d, fig_out)
        fig_landscape(d, fig_out, expected.get(ds_name(d)))

    if args.ablation:
        print("[ablation]")
        fig_ablation(args.ablation, expected)

    if not args.dirs and not args.ablation:
        ap.error("give result directories and/or --ablation ROOT")

    if len(args.dirs) >= 2:
        print("[comparison]")
        fig_comparison(args.dirs, args.out, expected)


if __name__ == "__main__":
    main()
