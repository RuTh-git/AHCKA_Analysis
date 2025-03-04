import config
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from data import data
import argparse
import random
import pandas as pd
import subprocess
from cluster import cluster

# Argument Parser
p = argparse.ArgumentParser(description='Set parameter')
p.add_argument('--data', type=str, default='coauthorship', help='Data type (coauthorship/cocitation)')
p.add_argument('--dataset', type=str, default='cora', help='Dataset name (e.g., cora/dblp for coauthorship, cora/citeseer for cocitation)')
p.add_argument('--tmax', type=int, default=200, help='t_max parameter')
p.add_argument('--seeds', type=int, default=0, help='Seed for randomness')
p.add_argument('--alpha', type=float, default=0.2, help='MHC parameter')
p.add_argument('--beta', type=float, default=0.5, help='Weight of KNN random walk')
p.add_argument('--knnk', type=int, default=10, help='K for KNN graph construction')
p.add_argument('--metric', type=bool, default=False, help='Calculate additional metrics: modularity')
p.add_argument('--rd_init', action='store_true', help='Initialize cluster labels randomly')
p.add_argument('--verbose', action='store_true', help='Print verbose logs')
p.add_argument('--scale', action='store_true', help='Use configurations for large-scale data')
p.add_argument('--interval', type=int, default=5, help='Interval between cluster predictions during orthogonal iterations')
p.add_argument('--method', type=str, default='knn', choices=['knn', 'hncut'], help='Select clustering method: knn (AHCKA) or hncut (Spectral Hypergraph Clustering)')
p.add_argument('--sensitivity', action='store_true', help='Run sensitivity analysis on k')

args = p.parse_args()

def run_clustering(k_value=None):
    """ Runs AHCKA or HNCut clustering """
    dataset = data.load(config.data, config.dataset)
    features = dataset['features_sp']
    labels = dataset['labels']

    labels = np.asarray(np.argmax(labels, axis=1)) if labels.ndim == 2 else labels
    config.labels = labels
    k = len(np.unique(labels))

    seed = config.seeds
    np.random.seed(seed)
    random.seed(seed)

    hg_adj = dataset['adj_sp']
    config.hg_adj = hg_adj
    config.features = features.copy()

    d_vec = np.asarray(config.hg_adj.sum(0)).flatten()
    deg_dict = {i: d_vec[i] for i in range(len(d_vec))}

    if k_value:
        config.knn_k = k_value  # Update k dynamically for sensitivity analysis

    if args.method == 'knn':
        results = cluster(hg_adj, features, k, deg_dict, alpha=config.alpha, beta=config.beta, tmax=config.tmax)
    elif args.method == 'hncut':
        results = cluster(hg_adj, features, k, deg_dict, method='hncut')

    return results

if __name__ == '__main__':
    config.data = args.data
    config.dataset = args.dataset
    config.metric = args.metric
    config.tmax = args.tmax
    config.beta = args.beta
    config.alpha = args.alpha
    config.seeds = args.seeds
    config.verbose = args.verbose
    config.cluster_interval = args.interval
    config.knn_k = args.knnk
    config.random_init = args.rd_init

    if args.scale:
        config.approx_knn = True
        config.init_iter = 1

    # If sensitivity analysis flag is passed, run for multiple k values
    if args.sensitivity:
        k_values = [2, 5, 10, 20, 50, 100, 200, 500, 1000]
        results = []

        for k in k_values:
            print(f"\nRunning AHCKA with k = {k}...")
            result = run_clustering(k_value=k)

            # Extract accuracy, NMI, ARI, runtime from results
            if result:
                acc, nmi, f1, ari, runtime, memory = result
                results.append([k, acc, f1, nmi, ari, runtime, memory])
        
        # Save results to CSV
        df = pd.DataFrame(results, columns=["k", "Accuracy", "F1-score", "NMI", "ARI", "Runtime", "Memory (MB)"])
        df.to_csv("sensitivity_k_results.csv", index=False)

        print("\nSensitivity analysis complete. Results saved to 'sensitivity_k_results.csv'.")
    else:
        # Run normally with selected method
        run_clustering()
