import pickle
import numpy as np
from scipy.sparse import lil_matrix
import os


def convert_hypergraph_to_hncut(input_root, output_root, subdir, suffix="_hncut"):
    data_dir = os.path.join(input_root, subdir)
    out_dir = os.path.join(output_root, subdir)
    os.makedirs(out_dir, exist_ok=True)

    # === Load data ===
    with open(os.path.join(data_dir, "hypergraph.pickle"), "rb") as f:
        hg_dict = pickle.load(f)
    with open(os.path.join(data_dir, "features.pickle"), "rb") as f:
        features = pickle.load(f)
    with open(os.path.join(data_dir, "labels.pickle"), "rb") as f:
        labels = pickle.load(f)

    # === Build mappings ===
    author_ids = list(hg_dict.keys())
    paper_ids = set(p for papers in hg_dict.values() for p in papers)

    author_to_idx = {author: i for i, author in enumerate(author_ids)}
    paper_to_idx = {paper: i for i, paper in enumerate(sorted(paper_ids))}

    num_authors = len(author_ids)
    num_papers = len(paper_to_idx)

    # === Construct incidence matrix ===
    H = lil_matrix((num_authors, num_papers), dtype=int)
    for author, papers in hg_dict.items():
        a_idx = author_to_idx[author]
        for paper in papers:
            p_idx = paper_to_idx[paper]
            H[a_idx, p_idx] = 1

    # === Remove singleton hyperedges and isolated nodes ===
    row_sums = np.array(H.sum(axis=1)).flatten()
    col_sums = np.array(H.sum(axis=0)).flatten()

    valid_nodes = np.where(row_sums > 0)[0]
    valid_edges = np.where(col_sums > 1)[0]

    H_cleaned = H[valid_nodes, :][:, valid_edges].tocsr()
    features_cleaned = features[valid_nodes]
    labels_cleaned = np.array(labels)[valid_nodes]  # 🩹 Fix applied here

    # === Save cleaned data ===
    with open(os.path.join(out_dir, f"hypergraph{suffix}.pickle"), "wb") as f:
        pickle.dump(H_cleaned, f)
    with open(os.path.join(out_dir, f"features{suffix}.pickle"), "wb") as f:
        pickle.dump(features_cleaned, f)
    with open(os.path.join(out_dir, f"labels{suffix}.pickle"), "wb") as f:
        pickle.dump(labels_cleaned, f)

    print(f"\n✅ Saved cleaned hypergraph to: {out_dir}")
    print(f"Original shape: {H.shape}, Cleaned shape: {H_cleaned.shape}")
    print(f"Removed {len(col_sums) - len(valid_edges)} singleton hyperedges")
    print(f"Retained {len(valid_nodes)} active nodes")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert hypergraph to HNCut-compatible format and save to hncut_data/")
    parser.add_argument("--input_root", type=str, default="data", help="Root input data folder")
    parser.add_argument("--output_root", type=str, default="hncut_data", help="Root output folder")
    parser.add_argument("--subdir", type=str, default="coauthorship/cora", help="Sub-directory path inside root")
    args = parser.parse_args()

    convert_hypergraph_to_hncut(args.input_root, args.output_root, args.subdir)
