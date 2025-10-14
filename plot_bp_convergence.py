# plot_bp_convergence.py
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_bp_stats(csv_path="bp_logs/stats.csv", outdir="bp_logs/plots"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"stats file not found: {csv_path}")
    os.makedirs(outdir, exist_ok=True)

    df = pd.read_csv(csv_path)

    # 1. Objective vs Bound
    plt.figure()
    plt.plot(df["iteration"], df["obj_value"], label="RMP obj")
    if "best_bound" in df.columns:
        plt.plot(df["iteration"], df["best_bound"], label="Best bound")
    plt.xlabel("Iteration")
    plt.ylabel("Objective")
    plt.legend()
    plt.title("Objective vs Bound")
    plt.savefig(f"{outdir}/obj_bound.png")

    # 2. Gap
    if "gap" in df.columns:
        plt.figure()
        plt.plot(df["iteration"], df["gap"] * 100, marker="o")
        plt.xlabel("Iteration")
        plt.ylabel("Gap (%)")
        plt.title("Gap curve")
        plt.savefig(f"{outdir}/gap.png")

    # 3. rc_min
    if "rc_min" in df.columns:
        plt.figure()
        plt.plot(df["iteration"], df["rc_min"], marker="o")
        plt.xlabel("Iteration")
        plt.ylabel("Min reduced cost")
        plt.title("Reduced cost convergence")
        plt.savefig(f"{outdir}/rc_min.png")

    # 4. Column growth
    if "num_columns" in df.columns and "num_new_columns" in df.columns:
        plt.figure()
        plt.plot(df["iteration"], df["num_columns"], label="Total cols")
        plt.bar(df["iteration"], df["num_new_columns"], alpha=0.5, label="New cols")
        plt.xlabel("Iteration")
        plt.ylabel("Columns")
        plt.legend()
        plt.title("Column growth")
        plt.savefig(f"{outdir}/columns.png")

    # 5. Coverage rate
    if "coverage_rate" in df.columns:
        plt.figure()
        plt.plot(df["iteration"], df["coverage_rate"] * 100, marker="o")
        plt.xlabel("Iteration")
        plt.ylabel("Coverage (%)")
        plt.title("Customer coverage")
        plt.savefig(f"{outdir}/coverage.png")

    # 6. Dual stats
    cols_dual = [c for c in ["avg_dual", "max_dual", "dual_entropy"] if c in df.columns]
    if cols_dual:
        fig, axes = plt.subplots(1, len(cols_dual), figsize=(5 * len(cols_dual), 4))
        if len(cols_dual) == 1:
            axes = [axes]
        for i, col in enumerate(cols_dual):
            axes[i].plot(df["iteration"], df[col], marker="o")
            axes[i].set_title(col)
            axes[i].set_xlabel("Iteration")
        plt.tight_layout()
        plt.savefig(f"{outdir}/duals.png")

    print(f"Plots saved in {outdir}/")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = "bp_logs/stats.csv"

    if len(sys.argv) > 2:
        outdir = sys.argv[2]
    else:
        outdir = "bp_logs/plots"

    plot_bp_stats(csv_path, outdir)
