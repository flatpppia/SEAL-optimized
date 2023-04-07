from __future__ import print_function
import numpy as np
import random
from tqdm import tqdm
import os, sys, pdb, math, time
import networkx as nx
import argparse
import scipy.io as sio
import scipy.sparse as ssp
from sklearn import metrics
from gensim.models import Word2Vec
import warnings
warnings.simplefilter('ignore', ssp.SparseEfficiencyWarning)
cur_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append('%s/../../pytorch_DGCNN' % cur_dir)
sys.path.append('%s/software/node2vec/src' % cur_dir)
from util import GNNGraph
import node2vec
import multiprocessing as mp
from itertools import islice


def load_npzdata(file_name, dataset_path='data/npzdata/'):
    """
    Load a graph from a Numpy binary file.
    :return: dict
    """
    with np.load(dataset_path+file_name, allow_pickle=True) as loader:
        loader = dict(loader)
        # print(loader.keys())
        ## dict_keys(['node_features', 'node_labels', 'edges', 'train_masks', 'val_masks', 'test_masks'])
        # print(loader['edges'].shape)
        # print(max(loader['node_labels']))
        adj_shape = loader['node_features'].shape[0]
        value = [1]*loader['edges'].shape[0]
        row = loader['edges'][:, 0]
        col = loader['edges'][:, 1]
        adj_matrix = ssp.csr_matrix((value, (row, col)), shape=[adj_shape, adj_shape])
        features = loader.get('node_features')  # 'numpy.ndarray'
        labels = loader.get('node_labels')
        graph = {'adj': adj_matrix, 'features': features, 'labels': labels}
        
        return graph

'''
def sample_link(net, h, label, all_unknown_as_negative=False):
    net_triu = ssp.triu(net, k=1)
    row, col, _ = ssp.find(net_triu)

    if h == 0.1:
        train_row, train_col, test_row, test_col = [], [], [], []
        for i in range(len(row)):
            if label[int(row[i])] != label[int(col[i])]:
                train_row.append(row[i])
                train_col.append(col[i])
            else:
                test_row.append(row[i])
                test_col.append(col[i])
        train_row = np.array(train_row)
        train_col = np.array(train_col)
        train_pos = (train_row, train_col)
        test_row = np.array(test_row)
        test_col = np.array(test_col)
        test_pos = (test_row, test_col)

        train_num = len(train_pos[0]) if train_pos else 0
        test_num = len(test_pos[0]) if test_pos else 0

        neg = ([], [])
        n = net.shape[0]
        print('sampling negative links for train and test')
        if not all_unknown_as_negative:
            # sample a portion unknown links as train_negs and test_negs (no overlap)
            while len(neg[0]) < train_num + test_num:
                i, j = random.randint(0, n - 1), random.randint(0, n - 1)
                if i < j and net[i, j] == 0:
                    neg[0].append(i)
                    neg[1].append(j)
                else:
                    continue
            train_neg = (neg[0][:train_num], neg[1][:train_num])
            test_neg = (neg[0][train_num:], neg[1][train_num:])
        else:
            # regard all unknown links as test_negs, sample a portion from them as train_negs
            while len(neg[0]) < train_num:
                i, j = random.randint(0, n - 1), random.randint(0, n - 1)
                if i < j and net[i, j] == 0:
                    neg[0].append(i)
                    neg[1].append(j)
                else:
                    continue
            train_neg = (neg[0], neg[1])
            test_neg_i, test_neg_j, _ = ssp.find(ssp.triu(net == 0, k=1))
            test_neg = (test_neg_i.tolist(), test_neg_j.tolist())

    elif h == 0.9:
        train_row, train_col, test_row, test_col = [], [], [], []
        for i in range(len(row)):
            if label[int(row[i])] == label[int(col[i])]:
                train_row.append(row[i])
                train_col.append(col[i])
            else:
                test_row.append(row[i])
                test_col.append(col[i])
        train_row = np.array(train_row)
        train_col = np.array(train_col)
        train_pos = (train_row, train_col)
        test_row = np.array(test_row)
        test_col = np.array(test_col)
        test_pos = (test_row, test_col)

        train_num = len(train_pos[0]) if train_pos else 0
        test_num = len(test_pos[0]) if test_pos else 0

        neg = ([], [])
        n = net.shape[0]
        print('sampling negative links for train and test')
        if not all_unknown_as_negative:
            # sample a portion unknown links as train_negs and test_negs (no overlap)
            while len(neg[0]) < train_num + test_num:
                i, j = random.randint(0, n - 1), random.randint(0, n - 1)
                if i < j and net[i, j] == 0:
                    neg[0].append(i)
                    neg[1].append(j)
                else:
                    continue
            train_neg = (neg[0][:train_num], neg[1][:train_num])
            test_neg = (neg[0][train_num:], neg[1][train_num:])
        else:
            # regard all unknown links as test_negs, sample a portion from them as train_negs
            while len(neg[0]) < train_num:
                i, j = random.randint(0, n - 1), random.randint(0, n - 1)
                if i < j and net[i, j] == 0:
                    neg[0].append(i)
                    neg[1].append(j)
                else:
                    continue
            train_neg = (neg[0], neg[1])
            test_neg_i, test_neg_j, _ = ssp.find(ssp.triu(net == 0, k=1))
            test_neg = (test_neg_i.tolist(), test_neg_j.tolist())
    return train_pos, train_neg, test_pos, test_neg
'''

def sample_neg(net, test_ratio=0.1, train_pos=None, test_pos=None, max_train_num=None, max_test_num=None, all_unknown_as_negative=False):
    net_triu = ssp.triu(net, k=1)  # get upper triangular matrix 上三角矩阵
    # sample positive links for train/test, class 'numpy.ndarray'
    row, col, _ = ssp.find(net_triu)  # row存储所有的行索引，col存储所有的列索引，(row[i],col[i])代表一个节点对
    # sample positive links if not specified
    print('Sampling positive links for train and test')
    if train_pos is None and test_pos is None:
        perm = random.sample(range(len(row)), len(row))  # 随即采样所有节点编号
        # exit(0)
        # random.shuffle(perm)  # 打乱顺序
        row, col = row[perm], col[perm]
        # print(row)
        split = int(math.ceil(len(row) * (1 - test_ratio)))  # ceil向上取整
        train_pos = (row[:split], col[:split])  # 以元组形式存储行、列两个列表
        test_pos = (row[split:], col[split:])
    # if max_train_num is set, randomly sample train links
    if max_train_num is not None and train_pos is not None:  # 不让连边数量过多
        perm = np.random.permutation(len(train_pos[0]))[:max_train_num]
        train_pos = (train_pos[0][perm], train_pos[1][perm])
    if max_test_num is not None:
        perm = np.random.permutation(len(test_pos[0]))[:max_test_num]
        test_pos = (test_pos[0][perm], test_pos[1][perm])
    # sample negative links for train/test
    train_num = len(train_pos[0]) if train_pos else 0
    test_num = len(test_pos[0]) if test_pos else 0
    neg = ([], [])
    n = net.shape[0]
    print('Sampling negative links for train and test')
    if not all_unknown_as_negative:
        # sample a portion unknown links as train_negs and test_negs (no overlap)
        while len(neg[0]) < train_num + test_num:
            i, j = random.randint(0, n-1), random.randint(0, n-1)
            if i < j and net[i, j] == 0:
                neg[0].append(i)
                neg[1].append(j)
            else:
                continue
        train_neg = (neg[0][:train_num], neg[1][:train_num])
        test_neg = (neg[0][train_num:], neg[1][train_num:])  # test_neg的数量与test_pos相等
    else:
        # regard all unknown links as test_negs, sample a portion from them as train_negs
        while len(neg[0]) < train_num:
            i, j = random.randint(0, n-1), random.randint(0, n-1)
            if i < j and net[i, j] == 0:
                neg[0].append(i)
                neg[1].append(j)
            else:
                continue
        train_neg = (neg[0], neg[1])
        test_neg_i, test_neg_j, _ = ssp.find(ssp.triu(net==0, k=1))
        test_neg = (test_neg_i.tolist(), test_neg_j.tolist())
    return train_pos, train_neg, test_pos, test_neg

    
def links2subgraphs(A, train_pos, train_neg, test_pos, test_neg, real_labels, n_p, h=1,
                    max_nodes_per_hop=None, node_information=None, no_parallel=False):
    # automatically select h from {1, 2}
    if h == 'auto':
        # split train into val_train and val_test
        _, _, val_test_pos, val_test_neg = sample_neg(A, 0.1)
        val_A = A.copy()
        val_A[val_test_pos[0], val_test_pos[1]] = 0
        val_A[val_test_pos[1], val_test_pos[0]] = 0
        val_auc_CN = CN(val_A, val_test_pos, val_test_neg)
        val_auc_AA = AA(val_A, val_test_pos, val_test_neg)
        print('\033[91mValidation AUC of AA is {}, CN is {}\033[0m'.format(val_auc_AA, val_auc_CN))
        if val_auc_AA >= val_auc_CN:
            h = 2
            print('\033[91mChoose h=2\033[0m')
        else:
            h = 1
            print('\033[91mChoose h=1\033[0m')

    # extract enclosing subgraphs
    max_n_label = {'value': 0}

    def helper(A, links, g_label, num_p):
        g_list = []
        if no_parallel:
            start = time.time()
            for i, j in tqdm(zip(links[0], links[1])):  # zip作用为将节点对组合为一个个元组, 并存进列表
                # 一个g对应一个节点对(i, j), 只要知道节点对就知道该连边的同质异质性，再在测试时做统计
                g, n_labels, n_features = subgraph_extraction_labeling((i, j), A, h, max_nodes_per_hop, node_information)
                max_n_label['value'] = max(max(n_labels), max_n_label['value'])
                g_list.append(GNNGraph(g, g_label, n_labels, n_features))
            end = time.time()
            print("Time eplased for subgraph extraction: {}s".format(end - start))
            return g_list
        else:
            # the parallel extraction code
            start = time.time()
            pool = mp.Pool(processes=num_p)  # 初始化并行处理
            # print("cpu最大进程数量：", mp.cpu_count())
            results = pool.map_async(parallel_worker,  # parallel_worker即多进程要处理的目标函数
                    [((i, j), A, real_labels, h, max_nodes_per_hop, node_information) for i, j in zip(links[0], links[1])])
            # map_async需要等待所有Task执行结束后返回list, 且按顺序等待Task的执行结果
            remaining = results._number_left
            pbar = tqdm(total=remaining)
            while True:
                pbar.update(remaining - results._number_left)
                if results.ready():
                    break
                remaining = results._number_left
                time.sleep(1)
            results = results.get()
            pool.close()
            pbar.close()
            g_list = [GNNGraph(g, g_label, link_label, n_labels, n_features) for g, link_label, n_labels, n_features in results]
            # print(g_list[0].num_edges) # print(g_list[0].edge_pairs)
            # print(type(g_list[0].edge_pairs))  # <class 'numpy.ndarray'>
            max_n_label['value'] = max(max([max(n_labels) for _, _, n_labels, _ in results]), max_n_label['value'])
            end = time.time()
            print("Time eplased for subgraph extraction: {}s".format(end-start))

            return g_list

    print('Enclosing subgraph extraction begins...')
    train_graphs, test_graphs = None, None
    if train_pos and train_neg:
        train_graphs1 = helper(A, train_pos, 1, n_p)
        train_graphs2 = helper(A, train_neg, 0, n_p)
        train_graphs = train_graphs1 + train_graphs2
    if test_pos and test_neg:
        test_graphs1 = helper(A, test_pos, 1, n_p)
        test_graphs2 = helper(A, test_neg, 0, n_p)
        test_graphs = test_graphs1 + test_graphs2
    elif test_pos:
        test_graphs = helper(A, test_pos, 1, n_p)
    return train_graphs, test_graphs, max_n_label['value']


def parallel_worker(x):
    return subgraph_extraction_labeling(*x)


def subgraph_extraction_labeling(ind, A, real_labels, h=1, max_nodes_per_hop=None, node_information=None):
    # extract the h-hop enclosing subgraph around link 'ind'
    """input: node pair 'ind', graph A, hop h"""
    # dist = 0
    nodes = set([ind[0], ind[1]])
    visited = set([ind[0], ind[1]])
    fringe = set([ind[0], ind[1]])
    nodes_dist = [0, 0]
    for dist in range(1, h+1):
        fringe = neighbors(fringe, A)
        fringe = fringe - visited
        visited = visited.union(fringe)
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(fringe):
                fringe = random.sample(fringe, max_nodes_per_hop)
        if len(fringe) == 0:
            break
        nodes = nodes.union(fringe)
        nodes_dist += [dist] * len(fringe)
    # move target nodes to top
    nodes.remove(ind[0])
    nodes.remove(ind[1])
    nodes = [ind[0], ind[1]] + list(nodes)
    subgraph = A[nodes, :][:, nodes]
    # subgraph:<class 'scipy.sparse._csc.csc_matrix'> 每个子图的节点已经重新编号好了
    # apply node-labeling
    labels = node_label(subgraph)
    # get node features
    features = None
    if node_information is not None:
        features = node_information[nodes]
    # construct nx graph
    g = nx.from_scipy_sparse_matrix(subgraph)  # <class 'networkx.classes.graph.Graph'>
    # remove link between target nodes
    if g.has_edge(0, 1):
        g.remove_edge(0, 1)

    if real_labels[ind[0]] == real_labels[ind[1]]:  # ind[0],ind[1]代表节点编号, real_label中是每个节点的label
        link_babel = 1  # link_babel代表每条目标连边的同质异质性
    else:
        link_babel = 0

    return g, link_babel, labels.tolist(), features


def neighbors(fringe, A):
    # find all 1-hop neighbors of nodes in fringe from A
    res = set()
    for node in fringe:
        nei, _, _ = ssp.find(A[:, node])
        nei = set(nei)
        res = res.union(nei)
    return res


def node_label(subgraph):
    # an implementation of the proposed double-radius node labeling (DRNL)
    K = subgraph.shape[0]
    subgraph_wo0 = subgraph[1:, 1:]
    subgraph_wo1 = subgraph[[0]+list(range(2, K)), :][:, [0]+list(range(2, K))]
    dist_to_0 = ssp.csgraph.shortest_path(subgraph_wo0, directed=False, unweighted=True)
    dist_to_0 = dist_to_0[1:, 0]
    dist_to_1 = ssp.csgraph.shortest_path(subgraph_wo1, directed=False, unweighted=True)
    dist_to_1 = dist_to_1[1:, 0]
    d = (dist_to_0 + dist_to_1).astype(int)
    d_over_2, d_mod_2 = np.divmod(d, 2)
    labels = 1 + np.minimum(dist_to_0, dist_to_1).astype(int) + d_over_2 * (d_over_2 + d_mod_2 - 1)
    labels = np.concatenate((np.array([1, 1]), labels))
    labels[np.isinf(labels)] = 0
    labels[labels > 1e6] = 0  # set inf labels to 0
    labels[labels < -1e6] = 0  # set -inf labels to 0
    return labels  # 这里的labels是个列表

    
def generate_node2vec_embeddings(A, emd_size=128, negative_injection=False, train_neg=None):
    if negative_injection:
        row, col = train_neg
        A = A.copy()
        A[row, col] = 1  # inject negative train
        A[col, row] = 1  # inject negative train
    nx_G = nx.from_scipy_sparse_matrix(A)
    G = node2vec.Graph(nx_G, is_directed=False, p=1, q=1)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(num_walks=10, walk_length=80)
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(walks, size=emd_size, window=10, min_count=0, sg=1, 
            workers=8, iter=1)
    wv = model.wv
    embeddings = np.zeros([A.shape[0], emd_size], dtype='float32')
    sum_embeddings = 0
    empty_list = []
    for i in range(A.shape[0]):
        if str(i) in wv:
            embeddings[i] = wv.word_vec(str(i))
            sum_embeddings += embeddings[i]
        else:
            empty_list.append(i)
    mean_embedding = sum_embeddings / (A.shape[0] - len(empty_list))
    embeddings[empty_list] = mean_embedding
    return embeddings


def AA(A, test_pos, test_neg):
    # Adamic-Adar score
    # print("A sum: ", A.sum(axis=1))
    A_ = A / np.log(A.sum(axis=1) + 1e-5)
    A_[np.isnan(A_)] = 0
    A_[np.isinf(A_)] = 0
    sim = A.dot(A_)
    return CalcAUC(sim, test_pos, test_neg)
    
        
def CN(A, test_pos, test_neg):
    # Common Neighbor score
    sim = A.dot(A)
    return CalcAUC(sim, test_pos, test_neg)


def CalcAUC(sim, test_pos, test_neg):
    pos_scores = np.asarray(sim[test_pos[0], test_pos[1]]).squeeze()
    neg_scores = np.asarray(sim[test_neg[0], test_neg[1]]).squeeze()
    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.hstack([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc
