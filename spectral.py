import numpy as np
import scipy.sparse as sp
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import accuracy_score, f1_score, normalized_mutual_info_score, adjusted_rand_score
import time
import psutil


def compute_hypergraph_laplacian(hg_adj):
    H = hg_adj
    D_v = np.array(H.sum(axis=1)).flatten()
    D_e = np.array(H.sum(axis=0)).flatten()

    D_v = np.clip(D_v, 1e-8, None)
    D_e = np.clip(D_e, 1e-8, None)

    Dv_inv = sp.diags(1.0 / D_v)
    De_inv = sp.diags(1.0 / D_e)

    L = sp.eye(H.shape[0]) - Dv_inv @ H @ De_inv @ H.T
    L = (L + L.T) / 2

    L.data[np.isnan(L.data)] = 0
    L.data[np.isinf(L.data)] = 0

    return L


def spectral_clustering(hg_adj, labels, k_clusters=6):
    start_time = time.time()
    process = psutil.Process()

    if hg_adj.shape[0] != len(labels):
        raise ValueError(f"Mismatch: labels={len(labels)} vs hg_adj rows={hg_adj.shape[0]}")

    L = compute_hypergraph_laplacian(hg_adj)

    if np.isnan(L.data).any() or np.isinf(L.data).any():
        raise ValueError("Laplacian still contains NaN or Inf values.")

    S = sp.eye(L.shape[0]) - L
    S = S.maximum(S.T)

    spectral = SpectralClustering(
        n_clusters=k_clusters,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=42
    )
    cluster_assignments = spectral.fit_predict(S)

    end_time = time.time()
    runtime = end_time - start_time
    memory_usage = process.memory_info().rss / (1024 * 1024)

    try:
        acc = accuracy_score(labels, cluster_assignments)
    except:
        acc = 0.0

    f1 = f1_score(labels, cluster_assignments, average='macro')
    nmi = normalized_mutual_info_score(labels, cluster_assignments)
    ari = adjusted_rand_score(labels, cluster_assignments)

    return acc, nmi, f1, ari, runtime, memory_usage


def discretize(eigenvectors):
    """
    Discretizes continuous eigenvectors to discrete cluster labels using KMeans.
    Each row in `eigenvectors` is treated as a feature vector.
    """
    n_clusters = eigenvectors.shape[1]
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(eigenvectors)
    return labels
