"""
Algorithm 1 Results Plotting Script
CS 4953 High Performance Machine Learning
Marisa Munoz, UTSA Spring 2026

Generates four figures from experimental results:
    Figure 1: Total training time per framework (1 GPU vs 2 GPU)
    Figure 2: Estimated energy consumption (avg power x total time)
    Figure 3: Final test accuracy per framework
    Figure 4: Test accuracy over epochs (learning curves, 2 GPU configs)

Usage:
    Run from the folder containing your CSV and power log files:
    python3 plot_results.py
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = os.path.expanduser("~/Desktop/algo1_results")

C = {
    "ddp_1":   "#2c7bb6",
    "ddp_2":   "#74add1",
    "tf_1":    "#d7191c",
    "tf_2":    "#fdae61",
    "axonn_1": "#1a9641",
    "axonn_2": "#a6d96a",
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
    """Load all CSV files matching a glob pattern. Returns a list of DataFrames."""
    files = sorted(glob.glob(os.path.join(DATA_DIR, pattern)))
    if not files:
        print(f"WARNING: no files found for pattern {pattern}")
    return [pd.read_csv(f) for f in files]


def total_time(runs):
    """
    Compute total training time across all epochs.
    Returns mean and standard deviation across multiple runs.
    """
    totals = [df["epoch_time_s"].sum() for df in runs]
    return np.mean(totals), np.std(totals)


def final_acc(runs):
    """
    Extract final epoch accuracy from each run.
    Handles both 'test_acc' (PyTorch/AxoNN) and 'val_acc' (TensorFlow) column names.
    Returns mean and standard deviation across runs.
    """
    accs = []
    for df in runs:
        if "test_acc" in df.columns:
            accs.append(df["test_acc"].iloc[-1])
        elif "val_acc" in df.columns:
            accs.append(df["val_acc"].iloc[-1])
    return np.mean(accs), np.std(accs)


def total_power_from_log(log_path):
    """
    Parse an nvidia-smi power log and return average total system power in Watts.

    nvidia-smi logs one row per GPU per polling interval, so power readings are
    summed across all GPUs at each timestamp to get total system draw. Only
    intervals where at least one GPU shows utilization above 5% are included,
    which filters out idle periods before and after training.
    """
    try:
        df = pd.read_csv(log_path, skipinitialspace=True)
        df.columns = [c.strip() for c in df.columns]
        pwr_col  = [c for c in df.columns if "power.draw" in c][0]
        util_col = [c for c in df.columns if "utilization.gpu" in c][0]
        ts_col   = [c for c in df.columns if "timestamp" in c][0]

        df[pwr_col]  = pd.to_numeric(
            df[pwr_col].astype(str).str.replace(" W", "").str.strip(), errors="coerce")
        df[util_col] = pd.to_numeric(
            df[util_col].astype(str).str.replace(" %", "").str.strip(), errors="coerce")

        power_per_ts = df.groupby(df[ts_col])[pwr_col].sum()
        util_per_ts  = df.groupby(df[ts_col])[util_col].max()

        active_ts    = util_per_ts[util_per_ts > 5].index
        active_power = power_per_ts[power_per_ts.index.isin(active_ts)]

        if active_power.empty:
            return power_per_ts.mean()
        return active_power.mean()

    except Exception as e:
        print(f"  Could not parse power log ({log_path}): {e}")
        return None


# Load all result files
ddp_1 = load_runs("results_pytorch_1gpu*.csv")
ddp_2 = load_runs("results_pytorch_2gpu*.csv")
tf_1  = load_runs("results_tensorflow_1gpu*.csv")
tf_2  = load_runs("results_tensorflow_2gpu*.csv")
ax_1  = load_runs("results_axonn_1gpu*.csv")
ax_2  = load_runs("results_axonn_2gpu*.csv")

configs = {
    "DDP 1 GPU":   ddp_1,
    "DDP 2 GPU":   ddp_2,
    "TF 1 GPU":    tf_1,
    "TF 2 GPU":    tf_2,
    "AxoNN 1 GPU": ax_1,
    "AxoNN 2 GPU": ax_2,
}

times, time_errs = {}, {}
accs,  acc_errs  = {}, {}

for name, runs in configs.items():
    if runs:
        t, te = total_time(runs)
        a, ae = final_acc(runs)
        times[name]     = t
        time_errs[name] = te
        accs[name]      = a
        acc_errs[name]  = ae
        print(f"{name:15s}  time={t:.1f}+/-{te:.1f}s  acc={a:.1f}+/-{ae:.1f}%")

# Read power logs and compute average total system power per configuration.
# For 1 GPU runs where no dedicated power log exists, power is estimated
# as half the measured 2 GPU total, which assumes symmetric GPU load.
p_ddp_1gpu  = total_power_from_log(os.path.join(DATA_DIR, "power_log_ddp_1gpu_r2.csv"))
p_ddp_2gpu  = total_power_from_log(os.path.join(DATA_DIR, "power_log_all.csv"))
p_tf_2gpu_a = total_power_from_log(os.path.join(DATA_DIR, "power_log_tf_2gpu.csv"))
p_tf_2gpu_b = total_power_from_log(os.path.join(DATA_DIR, "power_log_tf_2gpu_r2.csv"))
p_ax_2gpu_a = total_power_from_log(os.path.join(DATA_DIR, "power_log_axonn.csv"))
p_ax_2gpu_b = total_power_from_log(os.path.join(DATA_DIR, "power_log_axonn_r2.csv"))

p_tf_2gpu = np.mean([p_tf_2gpu_a, p_tf_2gpu_b])
p_ax_2gpu = np.mean([p_ax_2gpu_a, p_ax_2gpu_b])

power_map = {
    "DDP 1 GPU":   p_ddp_1gpu,
    "DDP 2 GPU":   p_ddp_2gpu,
    "TF 1 GPU":    p_tf_2gpu / 2,
    "TF 2 GPU":    p_tf_2gpu,
    "AxoNN 1 GPU": p_ax_2gpu / 2,
    "AxoNN 2 GPU": p_ax_2gpu,
}

# Energy = average power (W) x total training time (s)
energy = {}
for name in times:
    if name in power_map and power_map[name] is not None:
        energy[name] = power_map[name] * times[name]

print("\nEnergy estimates:")
for k, v in energy.items():
    print(f"  {k}: {power_map[k]:.0f}W x {times[k]:.1f}s = {v:.0f}J")

colors = [C["ddp_1"], C["ddp_2"], C["tf_1"], C["tf_2"], C["axonn_1"], C["axonn_2"]]


# Figure 1: Total training time
fig, ax = plt.subplots(figsize=(7, 4.2))
labels = list(times.keys())
vals   = [times[k] for k in labels]
errs   = [time_errs[k] for k in labels]

bars = ax.bar(labels, vals, color=colors[:len(labels)], width=0.55)
for i, (bar, val) in enumerate(zip(bars, vals)):
    ax.errorbar(bar.get_x() + bar.get_width() / 2, val,
                yerr=[[0], [errs[i]]], fmt='none', capsize=4,
                elinewidth=1, ecolor="#333")
    ax.text(bar.get_x() + bar.get_width() / 2, val + errs[i] + 1.5,
            f"{val:.0f}s", ha="center", va="bottom", fontsize=9, color="#333")

ax.set_ylabel("Total training time (s)")
ax.set_title("10-epoch training time on NVIDIA A40, ResNet18 / CIFAR-10",
             fontsize=10, pad=10)
ax.set_ylim(0, max(vals) * 1.18)
ax.tick_params(axis="x", labelsize=9)
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, "fig1_training_time.png"), bbox_inches="tight")
plt.close()
print("Saved fig1_training_time.png")


# Figure 2: Estimated energy consumption
# Log scale is used because TensorFlow's higher power draw produces values
# roughly one order of magnitude larger than PyTorch and AxoNN.
if energy:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    elabels = list(energy.keys())
    evals   = [energy[k] for k in elabels]

    bars = ax.bar(elabels, evals, color=colors[:len(elabels)], width=0.55)
    for bar, val in zip(bars, evals):
        ax.text(bar.get_x() + bar.get_width() / 2, val * 1.08,
                f"{val:.0f}J", ha="center", va="bottom", fontsize=8.5, color="#333")

    ax.set_yscale("log")
    ax.set_ylabel("Estimated energy (Joules, log scale)")
    ax.set_title("Estimated total energy, avg power draw x training time",
                 fontsize=10, pad=10)
    ax.tick_params(axis="x", labelsize=9)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "fig2_energy.png"), bbox_inches="tight")
    plt.close()
    print("Saved fig2_energy.png")


# Figure 3: Final test accuracy
fig, ax = plt.subplots(figsize=(7, 4.2))
alabels = list(accs.keys())
avals   = [accs[k] for k in alabels]
aerrs   = [acc_errs[k] for k in alabels]

bars = ax.bar(alabels, avals, color=colors[:len(alabels)], width=0.55)
for i, (bar, val) in enumerate(zip(bars, avals)):
    ax.errorbar(bar.get_x() + bar.get_width() / 2, val,
                yerr=[[0], [aerrs[i]]], fmt='none', capsize=4,
                elinewidth=1, ecolor="#333")
    ax.text(bar.get_x() + bar.get_width() / 2, val + aerrs[i] + 0.4,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=9, color="#333")

ax.set_ylabel("Test accuracy after 10 epochs (%)")
ax.set_title("Final test accuracy by framework and GPU count",
             fontsize=10, pad=10)
ax.set_ylim(60, 85)
ax.tick_params(axis="x", labelsize=9)
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, "fig3_accuracy.png"), bbox_inches="tight")
plt.close()
print("Saved fig3_accuracy.png")


# Figure 4: Learning curves, 2 GPU configurations only
# Shaded region shows +/- one standard deviation across the three runs.
fig, ax = plt.subplots(figsize=(7, 4.5))

curve_configs = [
    ("DDP 2 GPU",   ddp_2, C["ddp_2"],   "-"),
    ("TF 2 GPU",    tf_2,  C["tf_1"],    "--"),
    ("AxoNN 2 GPU", ax_2,  C["axonn_1"], "-."),
]

for label, runs, color, ls in curve_configs:
    if not runs:
        continue
    col      = "test_acc" if "test_acc" in runs[0].columns else "val_acc"
    mat      = np.array([df[col].values for df in runs])
    mean_acc = mat.mean(axis=0)
    std_acc  = mat.std(axis=0)
    epochs   = np.arange(1, len(mean_acc) + 1)

    ax.plot(epochs, mean_acc, color=color, linestyle=ls,
            linewidth=1.8, label=label, marker="o", markersize=3.5)
    ax.fill_between(epochs, mean_acc - std_acc, mean_acc + std_acc,
                    color=color, alpha=0.12)

ax.set_xlabel("Epoch")
ax.set_ylabel("Test accuracy (%)")
ax.set_title("Test accuracy over training, 2 GPU configurations",
             fontsize=10, pad=10)
ax.legend(frameon=False, fontsize=9)
ax.set_xlim(1, 10)
ax.set_xticks(range(1, 11))
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, "fig4_learning_curves.png"), bbox_inches="tight")
plt.close()
print("Saved fig4_learning_curves.png")

print("\nAll figures saved to", DATA_DIR)
