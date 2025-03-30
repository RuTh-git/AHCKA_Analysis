import os
import pickle
import numpy as np
import scipy.sparse as sp
import argparse

def remove_singleton_hyperedges(hg_adj):
    degrees = hg_adj.sum(axis=0).A1
    valid_edges = np.where(degrees > 1)[0]
    return hg_adj[:, valid_edges], valid_edges, len(degrees) - len(valid_edges)

def convert_to_hncut(input_root, output_root, subdir):
    data_dir = os.path.join(input_root, subdir)
    output_dir = os.path.join(output_root, subdir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n📦 Converting: {subdir}")

    npz_mode = os.path.exists(os.path.join(data_dir, "hypergraph.npz"))

    if npz_mode:
        hg = sp.load_npz(os.path.join(data_dir, "hypergraph.npz")).T  # Transpose here
        features = sp.load_npz(os.path.join(data_dir, "features.npz"))
        labels = np.load(os.path.join(data_dir, "labels.npy"))

        if hg.shape[0] != features.shape[0]:
            raise ValueError(f"Mismatch: hypergraph.shape[0]={hg.shape[0]} != features.shape[0]={features.shape[0]}")
    else:
        with open(os.path.join(data_dir, "hypergraph.pickle"), "rb") as f:
            hypergraph = pickle.load(f)
        with open(os.path.join(data_dir, "features.pickle"), "rb") as f:
            features = pickle.load(f)
        with open(os.path.join(data_dir, "labels.pickle"), "rb") as f:
            labels = pickle.load(f)

        n_nodes = features.shape[0]
        n_edges = len(hypergraph)
        rows, cols = [], []
        for j, edge in enumerate(hypergraph.values()):
            rows.extend(edge)
            cols.extend([j] * len(edge))
        data = np.ones(len(rows))
        hg = sp.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_edges))

    print(f"🔍 Original shape: {hg.shape}")

    hg_cleaned, valid_edges, removed = remove_singleton_hyperedges(hg)
    valid_nodes = np.where(hg_cleaned.sum(axis=1).A1 > 0)[0]
    hg_cleaned = hg_cleaned[valid_nodes, :]

    print(f"✅ Cleaned shape (after edge filtering): {hg_cleaned.shape}")
    print(f"❌ Removed {removed} singleton hyperedges")
    print(f"✔️ Retained {hg_cleaned.shape[0]} active nodes")

    if npz_mode:
        sp.save_npz(os.path.join(output_dir, "hypergraph_hncut.npz"), hg_cleaned)
        sp.save_npz(os.path.join(output_dir, "features_hncut.npz"), features[valid_nodes])
        np.save(os.path.join(output_dir, "labels_hncut.npy"), labels[valid_nodes])
    else:
        with open(os.path.join(output_dir, "hypergraph_hncut.pickle"), "wb") as f:
            pickle.dump(hg_cleaned, f)
        with open(os.path.join(output_dir, "features_hncut.pickle"), "wb") as f:
            pickle.dump(features[valid_nodes], f)
        with open(os.path.join(output_dir, "labels_hncut.pickle"), "wb") as f:
            pickle.dump(labels[valid_nodes], f)

    print(f"✅ Saved cleaned HNCut-compatible data to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert dataset to HNCut-compatible format.")
    parser.add_argument("--input_root", type=str, default="data", help="Root folder of datasets")
    parser.add_argument("--output_root", type=str, default="hncut_data", help="Root to save converted files")
    parser.add_argument("--subdir", type=str, required=True, help="Sub-directory inside input_root")

    args = parser.parse_args()
    convert_to_hncut(args.input_root, args.output_root, args.subdir)
