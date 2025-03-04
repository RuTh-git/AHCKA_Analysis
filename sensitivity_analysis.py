import pandas as pd
import matplotlib.pyplot as plt

# Load results from CSV
df = pd.read_csv("sensitivity_k_results.csv")

# Plot Clustering Quality Metrics
plt.figure(figsize=(10, 6))
plt.plot(df["k"], df["Accuracy"], label="Accuracy", marker="o")
plt.plot(df["k"], df["NMI"], label="NMI", marker="s")
plt.plot(df["k"], df["ARI"], label="ARI", marker="d")

plt.xscale("log")
plt.xlabel("k (log scale)")
plt.ylabel("Clustering Score")
plt.title("Impact of k on AHCKA Clustering Quality")
plt.legend()
plt.grid(True)
plt.savefig("clustering_quality.png")  # Save plot instead of showing it

# Plot Runtime and Memory Usage
plt.figure(figsize=(10, 6))
plt.plot(df["k"], df["Runtime"], label="Runtime (s)", marker="o", color="red")
plt.plot(df["k"], df["Memory (MB)"], label="Memory Usage (MB)", marker="s", color="purple")

plt.xscale("log")
plt.xlabel("k (log scale)")
plt.ylabel("Resource Consumption")
plt.title("Impact of k on AHCKA Computational Efficiency")
plt.legend()
plt.grid(True)
plt.savefig("computational_efficiency.png")  # Save plot instead of showing it

print("\nPlots saved as 'clustering_quality.png' and 'computational_efficiency.png'.")
