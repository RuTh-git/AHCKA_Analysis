import config
import numpy as np
import random
from data import data
from cluster import cluster
import argparse

# Argument Parser
p = argparse.ArgumentParser(description='Run AHCKA algorithm')
p.add_argument('--data', type=str, default='coauthorship', help='Data folder inside ./data/')
p.add_argument('--dataset', type=str, default='cora', help='Dataset name (e.g., cora)')
p.add_argument('--tmax', type=int, default=200, help='t_max parameter')
p.add_argument('--alpha', type=float, default=0.2, help='MHC parameter')
p.add_argument('--beta', type=float, default=0.5, help='Weight of KNN random walk')
p.add_argument('--knnk', type=int, default=10, help='K for KNN graph construction')
p.add_argument('--sensitivity', action='store_true', help='Run sensitivity analysis for k')
args = p.parse_args()

# Set config
config.data = args.data
config.dataset = args.dataset
config.tmax = args.tmax
config.alpha = args.alpha
config.beta = args.beta
config.knn_k = args.knnk

def run_ahcka(k_override=None):
    dataset = data.load("data", f"{args.data}/{args.dataset}")
    features = dataset['features_sp']
    labels = dataset['labels']
    labels = np.argmax(labels, axis=1) if labels.ndim == 2 else labels

    config.labels = labels
    k = len(np.unique(labels))
    config.hg_adj = dataset['adj_sp']
    config.features = features.copy()

    d_vec = np.asarray(config.hg_adj.sum(0)).flatten()
    deg_dict = {i: d_vec[i] for i in range(len(d_vec))}

    if k_override:
        config.knn_k = k_override

    return cluster(config.hg_adj, features, k, deg_dict,
                   alpha=config.alpha, beta=config.beta, tmax=config.tmax)

if __name__ == '__main__':
    if args.sensitivity:
        k_vals = [2, 5, 10, 20, 50, 100, 200, 500, 1000]
        dataset_obj = data.load("data", f"{args.data}/{args.dataset}")
        n_nodes = dataset_obj['features_sp'].shape[0]
        k_vals = [k for k in k_vals if k < n_nodes]

        import pandas as pd
        results = []
        for k in k_vals:
            print(f"\n🔍 AHCKA with k={k}")
            res = run_ahcka(k_override=k)
            if res:
                acc, nmi, f1, ari, time, mem = res
                print(f"Acc={acc:.3f}, F1={f1:.3f}, NMI={nmi:.3f}, ARI={ari:.3f}, Time={time:.3f}s, RAM={mem:.1f}MB")
                results.append([k, acc, f1, nmi, ari, time, mem])

        df = pd.DataFrame(results, columns=["k", "Accuracy", "F1", "NMI", "ARI", "Time", "Memory"])
        df.to_csv(f"sensitivity_ahcka_{args.data}_{args.dataset}.csv", index=False)
        print("✅ Saved sensitivity results.")
    else:
        res = run_ahcka()
        if res:
            acc, nmi, f1, ari, time, mem = res
            print(f"\n✅ AHCKA Results on {args.data}/{args.dataset}")
            print(f"Acc={acc:.3f}, F1={f1:.3f}, NMI={nmi:.3f}, ARI={ari:.3f}, Time={time:.3f}s, RAM={mem:.1f}MB")