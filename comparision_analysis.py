import pandas as pd
import matplotlib.pyplot as plt

# Load results from CSV
df = pd.read_csv("comparison_results.csv")

# Plot Clustering Quality Metrics
plt.figure(figsize=(10, 6))
plt.plot(df["Method"], df["Accuracy"], label="Accuracy", marker="o")
plt.plot(df["Method"], df["NMI"], label="NMI", marker="s")
plt.plot(df["Method"], df["ARI"], label="ARI", marker="d")

plt.xlabel("Method")
plt.ylabel("Clustering Score")
plt.title("Comparison of AHCKA vs HNCut Clustering Quality")
plt.legend()
plt.grid(True)
plt.show()

# Plot Runtime and Memory Usage
plt.figure(figsize=(10, 6))
plt.bar(df["Method"], df["Runtime"], label="Runtime (s)", color="red")
plt.bar(df["Method"], df["Memory (MB)"], label="Memory Usage (MB)", color="purple")

plt.xlabel("Method")
plt.ylabel("Resource Consumption")
plt.title("Comparison of AHCKA vs HNCut Computational Efficiency")
plt.legend()
plt.grid(True)
plt.show()
