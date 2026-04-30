"""
Algorithm 3 Results Plotting Script
CS 4953 High Performance Machine Learning
Marisa Munoz, UTSA Spring 2026

Generates 3 figures from Algorithm 3 ZeRO experimental results:
    Figure 1: Total training time per ZeRO stage vs DDP 2 GPU baseline
    Figure 2: Final test accuracy per ZeRO stage vs DDP 2 GPU baseline
    Figure 3: Test accuracy over epochs (learning curves)

Run from the folder containing your Algorithm 3 CSV files:
    python3 plot_algo3_results.py
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = os.path.expanduser("~/Desktop/algo3_results")

# DDP 2 GPU baseline from Algorithm 1
DDP_2GPU_TIME = 32.4
DDP_2GPU_ACC  = 75.5

C = {
    "stage1": "#2c7bb6",
    "stage2": "#1a9641",
    "stage3": "#d7191c",
    "ddp":    "#999999",
}

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.linewidth":   0.8,
    "figure.dpi":       150,
})


def load_runs(pattern):
    files = sorted(glob.glob(os.path.join(DATA_DIR, pattern)))
    if not files:
        print(f"WARNING: no files for {pattern}")
    return [pd.read_csv(f) for f in files]


def total_time(runs):
    totals = [df["epoch_time_s"].sum() for df in runs]
    return np.mean(totals), np.std(totals)


def final_acc(runs):
    accs = [df["test_acc"].iloc[-1] for df in runs]
    return np.mean(accs), np.std(accs)


stage1 = load_runs("results_zero_stage1_r*.csv")
stage2 = load_runs("results_zero_stage2_r*.csv")
stage3 = load_runs("results_zero_stage3_r*.csv")

configs = {
    "ZeRO Stage 1": stage1,
    "ZeRO Stage 2": stage2,
    "ZeRO Stage 3": stage3,
}

times, time_errs = {}, {}
accs,  acc_errs  = {}, {}

for name, runs in configs.items():
    if runs:
        t, te = total_time(runs)
        a, ae = final_acc(runs)
        times[name] = t
        time_errs[name] = te
        accs[name]  = a
        acc_errs[name] = ae
        print(f"{name:15s}  time={t:.1f}+/-{te:.1f}s  acc={a:.1f}+/-{ae:.1f}%")

colors = [C["stage1"], C["stage2"], C["stage3"]]


# Figure 1: Training time including DDP 2 GPU baseline
fig, ax = plt.subplots(figsize=(7, 4.2))

labels = ["DDP 2 GPU\n(baseline)"] + list(times.keys())
vals   = [DDP_2GPU_TIME] + [times[k] for k in times]
errs   = [0] + [time_errs[k] for k in time_errs]
bar_colors = [C["ddp"]] + colors

bars = ax.bar(labels, vals, color=bar_colors, width=0.55)
for i, (bar, val) in enumerate(zip(bars, vals)):
    ax.errorbar(bar.get_x() + bar.get_width()/2, val,
                yerr=[[0], [errs[i]]], fmt='none', capsize=4,
                elinewidth=1, ecolor="#333")
    ax.text(bar.get_x() + bar.get_width()/2, val + errs[i] + 1.5,
            f"{val:.0f}s", ha="center", va="bottom", fontsize=9, color="#333")

ax.set_ylabel("Total training time (s)")
ax.set_title("Training time by ZeRO stage vs DDP 2 GPU baseline\nNVIDIA A40, ResNet18 / CIFAR-10",
             fontsize=10, pad=10)
ax.set_ylim(0, max(vals) * 1.2)
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, "algo3_fig1_training_time.png"), bbox_inches="tight")
plt.close()
print("Saved algo3_fig1_training_time.png")


# Figure 2: Final accuracy vs DDP 2 GPU baseline
fig, ax = plt.subplots(figsize=(7, 4.2))

alabels = ["DDP 2 GPU\n(baseline)"] + list(accs.keys())
avals   = [DDP_2GPU_ACC] + [accs[k] for k in accs]
aerrs   = [0] + [acc_errs[k] for k in acc_errs]

bars = ax.bar(alabels, avals, color=bar_colors, width=0.55)
for i, (bar, val) in enumerate(zip(bars, avals)):
    ax.errorbar(bar.get_x() + bar.get_width()/2, val,
                yerr=[[0], [aerrs[i]]], fmt='none', capsize=4,
                elinewidth=1, ecolor="#333")
    ax.text(bar.get_x() + bar.get_width()/2, val + aerrs[i] + 0.2,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=9, color="#333")

ax.set_ylabel("Test accuracy after 10 epochs (%)")
ax.set_title("Final test accuracy by ZeRO stage vs DDP 2 GPU baseline",
             fontsize=10, pad=10)
ax.set_ylim(60, 85)
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, "algo3_fig2_accuracy.png"), bbox_inches="tight")
plt.close()
print("Saved algo3_fig2_accuracy.png")


# Figure 3: Learning curves
fig, ax = plt.subplots(figsize=(7, 4.5))

curve_configs = [
    ("ZeRO Stage 1", stage1, C["stage1"], "-",  1.2, "o", 2.5),
    ("ZeRO Stage 2", stage2, C["stage2"], "--", 1.2, "s", 2.5),
    ("ZeRO Stage 3", stage3, C["stage3"], "-.", 1.2, "^", 2.5),
]

for label, runs, color, ls, lw, marker, ms in curve_configs:
    if not runs:
        continue
    mat = np.array([df["test_acc"].values for df in runs])
    mean_acc = mat.mean(axis=0)
    std_acc  = mat.std(axis=0)
    epochs   = np.arange(1, len(mean_acc)+1)

    ax.plot(epochs, mean_acc, color=color, linestyle=ls,
            linewidth=lw, label=label, marker=marker, markersize=ms)
    ax.fill_between(epochs, mean_acc - std_acc, mean_acc + std_acc,
                    color=color, alpha=0.04)

ax.axhline(y=DDP_2GPU_ACC, color=C["ddp"], linestyle=":",
           linewidth=1.5, label="DDP 2 GPU baseline")

ax.set_xlabel("Epoch")
ax.set_ylabel("Test accuracy (%)")
ax.set_title("Test accuracy over training by ZeRO stage",
             fontsize=10, pad=10)
ax.legend(frameon=False, fontsize=9)
ax.set_xlim(1, 10)
ax.set_xticks(range(1, 11))
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, "algo3_fig3_learning_curves.png"), bbox_inches="tight")
plt.close()
print("Saved algo3_fig3_learning_curves.png")

print("\nAll figures saved to", DATA_DIR)
