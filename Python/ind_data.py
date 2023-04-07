import numpy as np
import scipy.sparse as sp
import warnings
import pickle as pkl
import sys
import networkx as nx
import time


class PlanetoidData:
    @staticmethod   # 定义静态方法
    def parse_index_file(filename):
        """Parse index file."""
        index = []
        for line in open(filename):
            index.append(int(line.strip()))
        return index

    @staticmethod
    def sample_mask(idx, l):
        """Create mask."""
        mask = np.zeros(l)
        mask[idx] = 1
        return np.array(mask, dtype=np.bool)

    @staticmethod
    def _pkl_load(f):
        if sys.version_info > (3, 0):
            return pkl.load(f, encoding='latin1')
        else:
            return pkl.load(f)

    @staticmethod
    def graphDict2Adj(graph):
        return nx.adjacency_matrix(nx.from_dict_of_lists(graph), nodelist=range(len(graph)))

    def getNXGraph(self):  # 定义实例方法
        G = nx.from_scipy_sparse_matrix(self.sparse_adj)
        for i, label in enumerate(self.labels):
            # To match the synthetic graph, label begins from 1.
            G.nodes[i]["color"] = int(label + 1)
        return G

    def load_data(self, dataset_str, dataset_path="data/data", save_plot=None, val_size=None):
        """
        Loads input data from gcn/data directory

        ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
        ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
        ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
            (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
        ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
        ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
        ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
        ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
            object;
        ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

        All objects above must be saved using python pickle module.

        :param dataset_str: Dataset name
        :return: All data input files loaded (as well the training/test data).
        """
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open("{}/{}.{}".format(dataset_path, dataset_str, names[i]), 'rb') as f:
                objects.append(self._pkl_load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = self.parse_index_file("{}/{}.test.index".format(dataset_path, dataset_str))
        test_idx_range = np.sort(test_idx_reorder)

        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        if len(test_idx_range_full) != len(test_idx_range):
            print(f"Patch for citeseer dataset is applied for dataset {dataset_str} at {dataset_path}")
            tx_extended[test_idx_range-min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range-min(test_idx_range), :] = ty
            ty = ty_extended
            self.non_valid_samples = set(test_idx_range_full) - set(test_idx_range)
        else:
            self.non_valid_samples = set()

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = self.graphDict2Adj(graph).astype(np.float32)

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        # Fix citeseer (and GeomGCN) bug
        self.non_valid_samples = self.non_valid_samples.union(set(list(np.where(labels.sum(1) == 0)[0])))

        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))

        train_mask = self.sample_mask(idx_train, labels.shape[0])
        test_mask = self.sample_mask(idx_test, labels.shape[0])
        val_mask = np.bitwise_not(np.bitwise_or(train_mask, test_mask))
        if val_size is not None:
            if np.sum(val_mask) > val_size:
                idx_val = range(len(y), len(y) + val_size)
                val_mask = self.sample_mask(idx_val, labels.shape[0])
            else:
                print(f"Val set size set to {np.sum(val_mask)} due to insufficient samples.")
        wild_mask = np.bitwise_not(train_mask + val_mask + test_mask)

        for n_i in self.non_valid_samples:
            if train_mask[n_i]:
                warnings.warn("Non valid samples detected in training set")
                train_mask[n_i] = False
            elif test_mask[n_i]:
                warnings.warn("Non valid samples detected in test set")
                test_mask[n_i] = False
            elif val_mask[n_i]:
                warnings.warn("Non valid samples detected in val set")
                val_mask[n_i] = False
            wild_mask[n_i] = False

        y_train = np.zeros(labels.shape)
        y_val = np.zeros(labels.shape)
        y_test = np.zeros(labels.shape)
        y_wild = np.zeros(labels.shape)
        y_train[train_mask, :] = labels[train_mask, :]
        y_val[val_mask, :] = labels[val_mask, :]
        y_test[test_mask, :] = labels[test_mask, :]
        y_wild[wild_mask, :] = labels[wild_mask, :]

        self._sparse_data["sparse_adj"] = adj
        self._sparse_data["features"] = features
        self._dense_data["y_all"] = labels
        self._dense_data["train_mask"] = train_mask
        self._dense_data["val_mask"] = val_mask
        self._dense_data["test_mask"] = test_mask
        self._dense_data["y_train"] = y_train
        self._dense_data["y_val"] = y_val
        self._dense_data["y_test"] = y_test
        self._dense_data["wild_mask"] = wild_mask
        self._dense_data["y_wild"] = y_wild
        self.__preprocessedAdj = None
        self.__preprocessedFeature = None

        return adj, features, y_train, y_val, y_test, y_wild, train_mask, val_mask, test_mask, wild_mask, labels

    def __getattribute__(self, name):
        if name in ("_sparse_data", "_dense_data"):
            return object.__getattribute__(self, name)
        elif name in self._sparse_data:
            return self._sparse_data[name]
        elif name in self._dense_data:
            return self._dense_data[name]
        else:
            return object.__getattribute__(self, name)

    def __setattr__(self, name, value):
        if name in ("_sparse_data", "_dense_data"):
            object.__setattr__(self, name, value)
        elif name in self._sparse_data:
            self._sparse_data[name] = value
        elif name in self._dense_data:
            self._dense_data[name] = value
        else:
            object.__setattr__(self, name, value)

    def __init__(self, dataset_str, dataset_path, val_size=None):
        self._sparse_data = dict()
        self._dense_data = dict()
        self.dataset_str = dataset_str
        self.dataset_path = dataset_path
        self.load_data(dataset_str, dataset_path, val_size=val_size)
        self._original_data = (self._sparse_data.copy(), self._dense_data.copy())


def count_link(adj, labels):
    dok = adj.todok()
    coo = dok.tocoo()
    row = coo.row  # 存储连边行坐标的列表
    col = coo.col  # 存储连边列坐标的列表
    linknum = len(row)
    tongzhi = []
    yizhi = []
    for i in range(linknum):
        if labels[row[i]] == labels[col[i]]:
            tongzhi.append((row[i], col[i]))
        else:
            yizhi.append((row[i], col[i]))
    tongzhi_num = len(tongzhi)
    h = tongzhi_num/linknum

    return tongzhi, yizhi


def setlabels(label_list):
    # 从大到小排序，l中第一个元组记录着值为1的元素的下标与该元素
    L = [sorted(enumerate(label_list[i]), key=lambda x: x[1], reverse=True)[0][0] for i in
         range(len(label_list))]
    L = np.array(L)  # 转成所需的形式
    return L


def main():
    dataset = PlanetoidData(dataset_str='ind.cora', dataset_path='data/data')
    adj, features, _, _, _, _, _, _, _, _, labels = dataset.load_data(dataset_str=dataset.dataset_str)
    single_labels = setlabels(labels)
    count_link(adj, single_labels)


if __name__ == '__main__':
    main()
