import os
import pickle
import numpy as np
import scipy.sparse as sp
import config


def load(data, dataset):
    """
    Load datasets in either .pickle or .npz format with optional _hncut suffix support.
    """
    suffix = "_hncut" if data.startswith("hncut_data") else ""

    # ✅ NPZ Support
    npz_path = os.path.join(data, dataset)
    if os.path.exists(os.path.join(npz_path, f"hypergraph{suffix}.npz")):
        hg_adj = sp.load_npz(os.path.join(npz_path, f"hypergraph{suffix}.npz"))
        np.clip(hg_adj.data, 0, 1, out=hg_adj.data)

        features = sp.load_npz(os.path.join(npz_path, f"features{suffix}.npz"))
        labels = np.load(os.path.join(npz_path, f"labels{suffix}.npy"))

        return {
            'features': features.todense(),
            'features_sp': features,
            'labels': labels,
            'n': features.shape[0],
            'e': hg_adj.shape[0],
            'name': dataset,
            'adj': hg_adj,
            'adj_sp': hg_adj
        }

    # ✅ Pickle Support
    ps = parser(data, dataset)  # Fixed instantiation
    parsed_data = ps.parse()    # Use .parse() to load

    hypergraph = parsed_data['hypergraph']
    features = parsed_data['features']
    labels = parsed_data['labels']

    if sp.issparse(hypergraph):
        adj_sp = hypergraph
    else:
        adj = np.zeros((len(hypergraph), features.shape[0]))
        for edge_idx, nodes in hypergraph.items():
            adj[edge_idx, nodes] = 1

        if config.remove_unconnected:
            nonzeros = adj.sum(0).nonzero()[0]
            adj = adj[:, nonzeros]
            features = features[nonzeros, :]
            labels = labels[nonzeros, :]
            pairs = adj.nonzero()
            hypergraph = {}
            for idx, edge in enumerate(pairs[0]):
                if edge not in hypergraph:
                    hypergraph[edge] = []
                hypergraph[edge].append(pairs[1][idx])

        adj_sp = sp.csr_matrix(adj)

    return {
        'features': features.todense() if sp.issparse(features) else features,
        'features_sp': features if sp.issparse(features) else sp.csr_matrix(features),
        'labels': labels,
        'n': features.shape[0],
        'e': adj_sp.shape[0],
        'name': dataset,
        'adj': adj_sp,
        'adj_sp': adj_sp
    }


class parser(object):
    def __init__(self, data, dataset):  # ✅ Fixed from _init_ to __init__
        import inspect
        current = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        project_root = os.path.dirname(current)
        self.d = os.path.join(project_root, data, dataset)
        self.data, self.dataset = data, dataset

    def parse(self):
        return self._load_data()

    def _load_data(self):
        with open(os.path.join(self.d, 'hypergraph.pickle'), 'rb') as f:
            hypergraph = pickle.load(f)
        with open(os.path.join(self.d, 'features.pickle'), 'rb') as f:
            features = pickle.load(f).todense()
        with open(os.path.join(self.d, 'labels.pickle'), 'rb') as f:
            labels = self._1hot(pickle.load(f))
        return {'hypergraph': hypergraph, 'features': features, 'labels': labels, 'n': features.shape[0]}

    def _1hot(self, labels):
        classes = sorted(set(labels))
        onehot = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
        return np.array(list(map(onehot.get, labels)), dtype=np.int32)
