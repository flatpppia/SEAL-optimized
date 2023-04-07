import torch
import numpy as np
import sys, copy, math, time, pdb
import pickle
import scipy.io as sio
import scipy.sparse as ssp
import os.path
import random
import argparse
from main import *
from util_functions import *
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader
from SAGPool.layers import SAGPool
import psutil
import torch.nn.functional as F
from ind_data import PlanetoidData
import twitch


def main():
    name = f'raw{"_cl"}'
    print(name)
    nn = f'{name}({len(name)})'
    print(nn)


def setlabels(label_list):
    # 从大到小排序，sorted结果中第一个元组记录着值为1的元素的下标与该元素
    L = [sorted(enumerate(label_list[i]), key=lambda x: x[1], reverse=True)[0][0] for i in range(len(label_list))]
    L = np.array(L)  # 转成所需的形式
    return L


def calcu_y_rate(row, col, nodelabels):
    yizhi = 0
    for i in range(len(row)):
        if nodelabels[row[i]] != nodelabels[col[i]]:
            yizhi += 1
    rate = yizhi/len(row)
    return rate

ind_dataset = PlanetoidData(dataset_str='ind.cora', dataset_path='data/data')
adj0, features, _, _, _, _, _, _, _, _, labels = ind_dataset.load_data(dataset_str=ind_dataset.dataset_str)
attributes0 = features.toarray()  # 'numpy.ndarrary'
single_labels0 = setlabels(labels)  # 记录每个节点label的一维列表，例如cora的label范围是0~6
print('nodes:', len(single_labels0))
net0 = adj0.tocsc(adj0)
net_triu0 = ssp.triu(net0, k=1)
row0, col0, _ = ssp.find(net_triu0)
print('edges:', len(row0))
rate0 = calcu_y_rate(row0, col0, single_labels0)
print(rate0)

G = load_npzdata('squirrel.npz')
adj1 = G['adj']
attributes1 = G['features']
single_labels1 = G['labels']
print('nodes:', len(single_labels1))
net1 = adj1.tocsc(adj1)
net_triu1 = ssp.triu(net1, k=1)
row1, col1, _ = ssp.find(net_triu1)
print('edges:', len(row1))
rate1 = calcu_y_rate(row1, col1, single_labels1)
print(rate1)

twitch_dataset = twitch.load_twitch_dataset('RU')
# print(dataset.graph['adj'])
adj2 = twitch_dataset.graph['adj']
attributes2 = twitch_dataset.graph['node_feat'].numpy()
single_labels2 = twitch_dataset.label.numpy()
print('nodes:', len(single_labels1))
net2 = adj2.tocsc(adj2)
net_triu2 = ssp.triu(net2, k=1)
row2, col2, _ = ssp.find(net_triu2)
print('edges:', len(row2))
rate2 = calcu_y_rate(row2, col2, single_labels2)
print(rate2)
