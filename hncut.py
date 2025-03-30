import argparse
import config
import numpy as np
from data import data
from spectral import spectral_clustering


def run_hncut():
    dataset = data.load(config.data, config.dataset)
    hg_adj = dataset["adj_sp"]
    labels = dataset["labels"]

    # Ensure labels are 1D array
    labels = np.asarray(np.argmax(labels, axis=1)) if labels.ndim == 2 else np.asarray(labels)
    config.labels = labels

    k = len(np.unique(labels))

    return spectral_clustering(hg_adj, labels, k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HNCut")
    parser.add_argument("--data", type=str, required=True, help="Root directory of the dataset")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset (relative to data root)")
    args = parser.parse_args()

    config.data = args.data
    config.dataset = args.dataset

    res = run_hncut()
    if res:
        acc, nmi, f1, ari, runtime, memory = res
        print(f"\n✅ HNCut Results on {args.dataset}")
        print(f"Acc={acc:.3f}, F1={f1:.3f}, NMI={nmi:.3f}, ARI={ari:.3f}, Time={runtime:.3f}s, RAM={memory:.1f}MB")
