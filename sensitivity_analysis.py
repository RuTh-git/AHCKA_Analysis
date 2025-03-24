import pandas as pd
import matplotlib.pyplot as plt
import os

# Updated list of datasets you're actually using
datasets = [
    "npz_query",
    "coauthorship_cora",
    "cocitation_cora",
    "cocitation_citeseer",
    "npz_20news"
]

# Create output folder for plots
os.makedirs("plots", exist_ok=True)

# Optional: track best k values for each dataset
summary = []

for dataset in datasets:
    filename = f"sensitivity_k_results_{dataset}.csv"
    try:
        df = pd.read_csv(filename)

        # Plot 1: Clustering Quality
        plt.figure(figsize=(10, 6))
        plt.plot(df["k"], df["Accuracy"], label="Accuracy", marker="o")
        plt.plot(df["k"], df["NMI"], label="NMI", marker="s")
        plt.plot(df["k"], df["ARI"], label="ARI", marker="d")
        plt.xscale("log")
        plt.xlabel("k (log scale)")
        plt.ylabel("Clustering Score")
        plt.title(f"Impact of k on AHCKA Clustering Quality ({dataset})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"plots/{dataset}_clustering_quality.png")
        plt.close()

        # Plot 2: Computational Efficiency
        plt.figure(figsize=(10, 6))
        plt.plot(df["k"], df["Runtime"], label="Runtime (s)", marker="o", color="red")
        plt.plot(df["k"], df["Memory (MB)"], label="Memory Usage (MB)", marker="s", color="purple")
        plt.xscale("log")
        plt.xlabel("k (log scale)")
        plt.ylabel("Resource Consumption")
        plt.title(f"Impact of k on AHCKA Computational Efficiency ({dataset})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"plots/{dataset}_computational_efficiency.png")
        plt.close()

        print(f"✔️ Plots saved for '{dataset}' in 'plots/' folder.")

        # Collect best k summary
        best_row = df.loc[df["Accuracy"].idxmax()]
        summary.append({
            "Dataset": dataset,
            "Best_k": int(best_row["k"]),
            "Accuracy": best_row["Accuracy"],
            "NMI": best_row["NMI"],
            "ARI": best_row["ARI"],
        })

    except FileNotFoundError:
        print(f"❌ File '{filename}' not found. Skipping...")

# Save summary if any
if summary:
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv("plots/k_sensitivity_summary.csv", index=False)
    print("\n📊 Summary of best k values saved to 'plots/k_sensitivity_summary.csv'")
