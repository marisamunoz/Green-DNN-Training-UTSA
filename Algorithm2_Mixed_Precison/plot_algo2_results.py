"""
Algorithm 2 Results Plotting Script
CS 4953 High Performance Machine Learning
Marisa Munoz, UTSA Spring 2026

Generates 3 figures from Algorithm 2 experimental results:
    Figure 1: Total training time per method
    Figure 2: Final test accuracy per method vs FP32 baseline
    Figure 3: Test accuracy over epochs (learning curves)

Run from the folder containing your Algorithm 2 CSV files:
    python3 plot_algo2_results.py
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = os.path.expanduser("~/Desktop/algo2_results")

# FP32 baseline from Algorithm 1 (DDP 1 GPU average across 3 runs)
FP32_BASELINE_ACC  = 78.42
FP32_BASELINE_TIME = 44.6

C = {
    "fp16":   "#2c7bb6",
    "bf16":   "#74add1",
    "int8":   "#d7191c",
    "distil": "#1a9641",
    "fp32":   "#999999",
}

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.linewidth":   0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "figure.dpi":       150,
})


def load_runs(pattern):
    files = sorted(glob.glob(os.path.join(DATA_DIR, pattern)))
    if not files:
        print(f"WARNING: no files found for pattern {pattern}")
    return [pd.read_csv(f) for f in files]


def total_time(runs):
    totals = [df["epoch_time_s"].sum() for df in runs]
    return np.mean(totals), np.std(totals)


def final_acc(runs):
    accs = []
    for df in runs:
        if "test_acc" in df.columns:
            # For INT8, get the quantized row (phase == INT8_quantized)
            if "phase" in df.columns:
                quant_rows = df[df["phase"] == "INT8_quantized"]
                if not quant_rows.empty:
                    accs.append(quant_rows["test_acc"].iloc[-1])
                    continue
            accs.append(df["test_acc"].iloc[-1])
    return np.mean(accs), np.std(accs)


def fp32_training_time(runs):
    """For INT8, only count the FP32 training phase time."""
    totals = []
    for df in runs:
        if "phase" in df.columns:
            fp32_rows = df[df["phase"] == "FP32_training"]
            totals.append(fp32_rows["epoch_time_s"].sum())
        else:
            totals.append(df["epoch_time_s"].sum())
    return np.mean(totals), np.std(totals)


# Load all runs
fp16  = load_runs("results_fp16_r*.csv")
bf16  = load_runs("results_bf16_r*.csv")
int8  = load_runs("results_int8_r*.csv")
distil = load_runs("results_distillation_r*.csv")

configs = {
    "FP16":         fp16,
    "BF16":         bf16,
    "INT8":         int8,
    "Distillation": distil,
}

times, time_errs = {}, {}
accs,  acc_errs  = {}, {}

for name, runs in configs.items():
    if runs:
        if name == "INT8":
            t, te = fp32_training_time(runs)
        else:
            t, te = total_time(runs)
        a, ae = final_acc(runs)
        times[name] = t
        time_errs[name] = te
        accs[name]  = a
        acc_errs[name] = ae
        print(f"{name:15s}  time={t:.1f}+/-{te:.1f}s  acc={a:.1f}+/-{ae:.1f}%")

colors = [C["fp16"], C["bf16"], C["int8"], C["distil"]]


# Figure 1: Training time including FP32 baseline
fig, ax = plt.subplots(figsize=(7, 4.2))

labels = ["FP32\n(baseline)"] + list(times.keys())
vals   = [FP32_BASELINE_TIME] + [times[k] for k in times]
errs   = [0] + [time_errs[k] for k in time_errs]
bar_colors = [C["fp32"]] + colors

bars = ax.bar(labels, vals, color=bar_colors, width=0.55)
for i, (bar, val) in enumerate(zip(bars, vals)):
    ax.errorbar(bar.get_x() + bar.get_width()/2, val,
                yerr=[[0], [errs[i]]], fmt='none', capsize=4,
                elinewidth=1, ecolor="#333")
    ax.text(bar.get_x() + bar.get_width()/2, val + errs[i] + 1.0,
            f"{val:.0f}s", ha="center", va="bottom", fontsize=9, color="#333")

ax.set_ylabel("Total training time (s)")
ax.set_title("Training time by precision method, NVIDIA A40, ResNet18 / CIFAR-10",
             fontsize=10, pad=10)
ax.set_ylim(0, max(vals) * 1.2)
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, "algo2_fig1_training_time.png"), bbox_inches="tight")
plt.close()
print("Saved algo2_fig1_training_time.png")


# Figure 2: Final accuracy vs FP32 baseline
fig, ax = plt.subplots(figsize=(7, 4.2))

alabels = ["FP32\n(baseline)"] + list(accs.keys())
avals   = [FP32_BASELINE_ACC] + [accs[k] for k in accs]
aerrs   = [0] + [acc_errs[k] for k in acc_errs]

bars = ax.bar(alabels, avals, color=bar_colors, width=0.55)
for i, (bar, val) in enumerate(zip(bars, avals)):
    ax.errorbar(bar.get_x() + bar.get_width()/2, val,
                yerr=[[0], [aerrs[i]]], fmt='none', capsize=4,
                elinewidth=1, ecolor="#333")
    ax.text(bar.get_x() + bar.get_width()/2, val + aerrs[i] + 0.2,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=9, color="#333")

ax.set_ylabel("Test accuracy after 10 epochs (%)")
ax.set_title("Final test accuracy by precision method vs FP32 baseline",
             fontsize=10, pad=10)
ax.set_ylim(70, 85)
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, "algo2_fig2_accuracy.png"), bbox_inches="tight")
plt.close()
print("Saved algo2_fig2_accuracy.png")


# Figure 3: Learning curves
fig, ax = plt.subplots(figsize=(7, 4.5))

curve_configs = [
    ("FP16",         fp16,   C["fp16"],   "-",   1.2,  "o",  2.5),
    ("BF16",         bf16,   C["bf16"],   "--",  1.2,  "s",  2.5),
    ("INT8",         int8,   C["int8"],   "-",   1.2,  "^",  2.5),
    ("Distillation", distil, C["distil"], "-.",  1.2,  "D",  2.5),
]

for label, runs, color, ls, lw, marker, ms in curve_configs:
    if not runs:
        continue

    if label == "INT8":
        fp32_runs = []
        for df in runs:
            if "phase" in df.columns:
                fp32_runs.append(df[df["phase"] == "FP32_training"].reset_index(drop=True))
            else:
                fp32_runs.append(df)
        mat = np.array([df["test_acc"].values for df in fp32_runs])
        mean_acc = mat.mean(axis=0)
        std_acc  = mat.std(axis=0)
        epochs   = np.arange(1, len(mean_acc)+1)

        ax.plot(epochs, mean_acc, color=color, linestyle=ls,
                linewidth=lw, label="INT8 (FP32 training phase)", marker=marker, markersize=ms)
        ax.fill_between(epochs, mean_acc - std_acc, mean_acc + std_acc,
                        color=color, alpha=0.07)

        int8_final = accs["INT8"]
        ax.plot(10, int8_final, marker="*", markersize=14, color=color,
                zorder=5, label="INT8 quantized (final)")
    else:
        col = "test_acc"
        mat = np.array([df[col].values for df in runs])
        mean_acc = mat.mean(axis=0)
        std_acc  = mat.std(axis=0)
        epochs   = np.arange(1, len(mean_acc)+1)

        ax.plot(epochs, mean_acc, color=color, linestyle=ls,
                linewidth=lw, label=label, marker=marker, markersize=ms)
        ax.fill_between(epochs, mean_acc - std_acc, mean_acc + std_acc,
                        color=color, alpha=0.07)

# Add FP32 baseline as horizontal reference line
ax.axhline(y=FP32_BASELINE_ACC, color=C["fp32"], linestyle=":",
           linewidth=1.5, label="FP32 baseline")

ax.set_xlabel("Epoch")
ax.set_ylabel("Test accuracy (%)")
ax.set_title("Test accuracy over training by precision method",
             fontsize=10, pad=10)
ax.legend(frameon=False, fontsize=9)
ax.set_xlim(1, 10)
ax.set_xticks(range(1, 11))
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, "algo2_fig3_learning_curves.png"), bbox_inches="tight")
plt.close()
print("Saved algo2_fig3_learning_curves.png")

print("\nAll figures saved to", DATA_DIR)
