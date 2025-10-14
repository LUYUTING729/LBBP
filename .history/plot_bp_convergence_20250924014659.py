# plot_bp_convergence.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_bp_stats(csv_path="bp_logs/stats.csv", outdir="bp_logs/plots"):
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(csv_path)

    # 1. obj & bound
    plt.figure()
    plt.plot(df["iteration"], df["obj_value"], label="RMP obj")
    plt.plot(df["iteration"], df["best_bound"], label="Best bound")
    plt.xlabel("Iteration")
    plt.ylabel("Objective")
    plt.legend()
    plt.title("Objective vs Bound")
    plt.savefig(f"{outdir}/obj_bound.png")

    # 2. gap
    plt.figure()
    plt.plot(df["iteration"], df["gap"] * 100, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Gap (%)")
    plt.title("Gap curve")
    plt.savefig(f"{outdir}/gap.png")

    # 3. rc_min
    plt.figure()
    plt.plot(df["iteration"], df["rc_min"], marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Min reduced cost")
    plt.title("Reduced cost convergence")
    plt.savefig(f"{outdir}/rc_min.png")

    # 4. column growth
    plt.figure()
    plt.plot(df["iteration"], df["num_columns"], label="Total cols")
    plt.bar(df["iteration"], df["num_new_columns"], alpha=0.5, label="New cols")
    plt.xlabel("Iteration")
    plt.ylabel("Columns")
    plt.legend()
    plt.title("Column growth")
    plt.savefig(f"{outdir}/columns.png")

    # 5. coverage
    plt.figure()
    plt.plot(df["iteration"], df["coverage_rate"] * 100, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Coverage (%)")
    plt.title("Customer coverage")
    plt.savefig(f"{outdir}/coverage.png")

    # 6. dual stats
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(df["iteration"], df["avg_dual"], marker="o")
    axes[0].set_title("Average dual")
    axes[1].plot(df["iteration"], df["max_dual"], marker="o")
    axes[1].set_title("Max dual")
    axes[2].plot(df["iteration"], df["dual_entropy"], marker="o")
    axes[2].set_title("Dual entropy")
    for ax in axes:
        ax.set_xlabel("Iteration")
    plt.tight_layout()
    plt.savefig(f"{outdir}/duals.png")

    print(f"Plots saved in {outdir}/")

if __name__ == "__main__":
    plot_bp_stats()
