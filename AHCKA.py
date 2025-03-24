import config
import numpy as np
import scipy.sparse as sp
from data import data
import argparse
import random
import pandas as pd
import subprocess
from cluster import cluster
from spectral import spectral_clustering  # Import HNCut function
import os

# Argument Parser
p = argparse.ArgumentParser(description='Set parameter')
p.add_argument('--data', type=str, default='coauthorship', help='Data type (coauthorship/cocitation/npz)')
p.add_argument('--dataset', type=str, default='cora', help='Dataset name (e.g., cora/dblp/citeseer/20news)')
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
        results = spectral_clustering(hg_adj, k)  # Call spectral clustering

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

    # Run sensitivity analysis if flag is enabled
    if args.sensitivity:
        dataset_obj = data.load(config.data, config.dataset)
        n_nodes = dataset_obj['features_sp'].shape[0]

        base_k_values = [2, 5, 10, 20, 50, 100, 200, 500, 1000]
        k_values = [k for k in base_k_values if k < n_nodes]

        results = []

        for k in k_values:
            print(f"\nRunning AHCKA on {args.dataset} with k = {k}...")
            result = run_clustering(k_value=k)

            if result:
                acc, nmi, f1, ari, runtime, memory = result
                print(f"Acc={acc:.3f} F1={f1:.3f} NMI={nmi:.3f} ARI={ari:.3f} Time={runtime:.3f}s RAM={memory}MB")
                results.append([k, acc, f1, nmi, ari, runtime, memory])

        df = pd.DataFrame(results, columns=["k", "Accuracy", "F1-score", "NMI", "ARI", "Runtime", "Memory (MB)"])
        output_path = f"sensitivity_k_results_{args.data}_{args.dataset}.csv"
        df.to_csv(output_path, index=False)

        print(f"\n✅ Sensitivity analysis complete. Results saved to '{output_path}'.")

    else:
        # Run normally with selected method (AHCKA or HNCut)
        run_clustering()
