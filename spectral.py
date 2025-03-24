import numpy as np
import scipy.sparse as sp
from sklearn.cluster import SpectralClustering
from sklearn.utils import check_random_state, as_float_array
from sklearn.metrics import accuracy_score, f1_score, normalized_mutual_info_score, adjusted_rand_score
import time
import psutil
from config import labels  # Ground truth labels used for evaluation


def discretize(vectors, copy=True, max_svd_restarts=3, n_iter_max=50, random_state=None):
    """
    Converts continuous spectral embeddings into discrete clustering assignments using an iterative 
    rotation method to minimize the normalized cut.
    """
    from scipy.sparse import csc_matrix
    from scipy.linalg import LinAlgError

    random_state = check_random_state(random_state)
    vectors = as_float_array(vectors, copy=copy)
    eps = np.finfo(float).eps
    n_samples, n_components = vectors.shape

    norm_ones = np.sqrt(n_samples)
    for i in range(n_components):
        vectors[:, i] = (vectors[:, i] / np.linalg.norm(vectors[:, i])) * norm_ones
        if vectors[0, i] != 0:
            vectors[:, i] = -1 * vectors[:, i] * np.sign(vectors[0, i])

    vectors = vectors / np.sqrt((vectors ** 2).sum(axis=1))[:, np.newaxis]
    vectors[np.isnan(vectors)] = 0
    vectors[np.isinf(vectors)] = 0

    svd_restarts = 0
    has_converged = False

    while (svd_restarts < max_svd_restarts) and not has_converged:
        rotation = np.identity(n_components)
        last_objective_value = 0.0
        n_iter = 0
        max_label = None

        while not has_converged:
            n_iter += 1
            t_discrete = np.dot(vectors, rotation)
            labels = t_discrete.argmax(axis=1)
            vectors_discrete = csc_matrix(
                (np.ones(len(labels)), (np.arange(0, n_samples), labels)),
                shape=(n_samples, n_components)).toarray()
            
            vectors_f = vectors_discrete
            vectors_fs = np.sqrt(vectors_f.sum(axis=0))
            vectors_fs[vectors_fs == 0] = 1
            vectors_f = vectors_f * 1.0 / vectors_fs

            t_svd = vectors_f.T.dot(vectors)

            try:
                U, S, Vh = np.linalg.svd(t_svd)
                svd_restarts += 1
            except LinAlgError:
                print("SVD did not converge")
                break

            ncut_value = 2.0 * (n_samples - S.sum())
            if ncut_value > 0:
                max_label = labels
            if ((abs(ncut_value - last_objective_value) < eps) or (n_iter > n_iter_max)):
                has_converged = True
            else:
                last_objective_value = ncut_value
                rotation = np.dot(Vh.T, U.T)

    if not has_converged:
        raise LinAlgError('SVD did not converge')

    return max_label, vectors_f


def compute_hypergraph_laplacian(hg_adj):
    """
    Computes the Hypergraph Laplacian for spectral clustering.
    Handles isolated nodes/hyperedges safely.
    """
    # Compute degree vectors
    dv = np.array(hg_adj.sum(axis=1)).flatten()
    de = np.array(hg_adj.sum(axis=0)).flatten()

    # Add epsilon to avoid division by zero
    dv[dv == 0] = 1e-8
    de[de == 0] = 1e-8

    # Construct diagonal matrices
    D_v = np.diag(dv)
    D_e_inv = np.diag(1.0 / de)

    # Compute Laplacian
    L = D_v - hg_adj @ D_e_inv @ hg_adj.T
    return L



def spectral_clustering(hg_adj, k_clusters=6):
    """
    Performs spectral clustering using HNCut (Hypergraph Normalized Cut).

    Parameters:
    - hg_adj: Hypergraph adjacency matrix
    - k_clusters: Number of clusters

    Returns:
    - acc, nmi, f1, ari, runtime, memory_usage
    """
    start_time = time.time()
    process = psutil.Process()

    # Compute Laplacian
    L = compute_hypergraph_laplacian(hg_adj)

    # Spectral Clustering
    spectral = SpectralClustering(n_clusters=k_clusters, affinity="precomputed", assign_labels="kmeans")
    cluster_assignments = spectral.fit_predict(L)

    end_time = time.time()
    runtime = end_time - start_time
    memory_usage = process.memory_info().rss / (1024 * 1024)  # MB

    # Evaluate metrics
    y_true = np.array(labels)
    try:
        acc = accuracy_score(y_true, cluster_assignments)
    except:
        acc = 0.0

    f1 = f1_score(y_true, cluster_assignments, average='macro')
    nmi = normalized_mutual_info_score(y_true, cluster_assignments)
    ari = adjusted_rand_score(y_true, cluster_assignments)

    return acc, nmi, f1, ari, runtime, memory_usage
