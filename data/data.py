import os
import pickle
import numpy as np
import scipy.sparse as sp
import config

def load(data, dataset):
    if data == 'npz':
        return load_npz(dataset)

    ps = parser(data, dataset)
    hncut_mode = data.startswith('hncut_data')

    suffix = '_hncut' if hncut_mode else ''

    # Load files with or without suffix
    with open(os.path.join(ps.d, f'hypergraph{suffix}.pickle'), 'rb') as handle:
        hypergraph = pickle.load(handle)

    with open(os.path.join(ps.d, f'features{suffix}.pickle'), 'rb') as handle:
        features = pickle.load(handle)

    with open(os.path.join(ps.d, f'labels{suffix}.pickle'), 'rb') as handle:
        raw_labels = pickle.load(handle)
        labels = ps._1hot(raw_labels) if not isinstance(raw_labels, np.ndarray) or raw_labels.ndim == 1 else raw_labels

    # If already sparse matrix, treat as adjacency
    if sp.issparse(hypergraph):
        adj_sp = hypergraph
    else:
        adj = np.zeros((len(hypergraph), features.shape[0]))
        for index, edge in enumerate(hypergraph):
            hypergraph[edge] = list(hypergraph[edge])
            adj[index, hypergraph[edge]] = 1

        if config.remove_unconnected:
            nonzeros = adj.sum(0).nonzero()[0]
            adj = adj[:, nonzeros]
            features = features[nonzeros, :]
            labels = labels[nonzeros, :]
            pairs = adj.nonzero()
            hypergraph = {}
            for index, edge in enumerate(pairs[0]):
                if edge not in hypergraph:
                    hypergraph[edge] = []
                hypergraph[edge].append(pairs[1][index])

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

def load_npz(dataset):
    hg_adj = sp.load_npz(f'data/npz/{dataset}/hypergraph.npz')
    np.clip(hg_adj.data, 0, 1, out=hg_adj.data)
    features = sp.load_npz(f'data/npz/{dataset}/features.npz')
    labels = np.load(f'data/npz/{dataset}/labels.npy')
    return {
        'features_sp': features,
        'labels': labels,
        'n': features.shape[0],
        'e': hg_adj.shape[0],
        'name': dataset,
        'adj': hg_adj,
        'adj_sp': hg_adj
    }

class parser(object):
    def __init__(self, data, dataset):
        import inspect
        current = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        project_root = os.path.dirname(current)  # go one level up
        self.d = os.path.join(project_root, data, dataset)
        self.data, self.dataset = data, dataset

    def parse(self):
        name = "_load_data"
        function = getattr(self, name, lambda: {})
        return function()

    def _load_data(self):
        with open(os.path.join(self.d, 'hypergraph.pickle'), 'rb') as handle:
            hypergraph = pickle.load(handle)

        with open(os.path.join(self.d, 'features.pickle'), 'rb') as handle:
            features = pickle.load(handle).todense()

        with open(os.path.join(self.d, 'labels.pickle'), 'rb') as handle:
            labels = self._1hot(pickle.load(handle))

        return {'hypergraph': hypergraph, 'features': features, 'labels': labels, 'n': features.shape[0]}

    def _1hot(self, labels):
        classes = sorted(set(labels))
        onehot = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
        return np.array(list(map(onehot.get, labels)), dtype=np.int32)
