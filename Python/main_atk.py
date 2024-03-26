import torch
import numpy as np
import sys, copy, math, time, pdb
import pickle
import scipy.io as sio
import scipy.sparse as ssp
import os
import random
import argparse
from main import *
from util_functions import *
# from dataset import CustomDataset
from ind_data import PlanetoidData
# from torch.utils.data import DataLoader
# sys.path.append('%s/../../pytorch_DGCNN' % os.path.dirname(os.path.realpath(__file__)))
import psutil
import twitch
import multiprocessing as mp
import networkx as nx
from homo_graph import *
from deeprobust.graph.global_attack import DICE

parser = argparse.ArgumentParser(description='Link Prediction with SEAL')
# general settings
parser.add_argument('--d-name', default='cornell', help='data name')
# ind.cora、citeseer、pubmed
# actor、squirrel、chameleon、wisconsin、cornell、texas.npz
parser.add_argument('--train-name', default=None, help='train name')
parser.add_argument('--test-name', default=None, help='test name')
parser.add_argument('--only-predict', action='store_true', default=False,
                    help='if True, will load the saved model and output predictions\
                    for links in test-name; you still need to specify train-name\
                    in order to build the observed network and extract subgraphs')
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--max-train', type=int, default=None,
                    help='set maximum number of train links (to fit into memory)')
parser.add_argument('--max-test', type=int, default=None,
                    help='set maximum number of test links (to fit into memory)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--test-ratio', type=float, default=0.1, help='ratio of test links')
parser.add_argument('--no-parallel', action='store_true', default=False,
                    help='if True, use single thread for subgraph extraction; \
                    by default use all cpu cores to extract subgraphs in parallel')
parser.add_argument('--all-unknown-as-negative', action='store_true', default=False,
                    help='if True, regard all unknown links as negative test data; \
                    sample a portion from them as negative training data. Otherwise,\
                    train negative and test negative data are both sampled from unknown links without overlap.')
# model settings
parser.add_argument('--hop', default='2', metavar='S', help='enclosing subgraph hop number, options: 1, 2,..."auto"')
parser.add_argument('--max-nodes-per-hop', type=int, default=64, help='upper bound the nodes per hop by subsampling')
parser.add_argument('--use-embedding', action='store_true', default=False, help='whether to use node2vec')
parser.add_argument('--use-attribute', default=False, help='whether to use node attributes')
parser.add_argument('--save-model', default=False, help='save the final model')
parser.add_argument('--hh', type=float, default=0.1, help='the homophily percent of syn-cora')
parser.add_argument('--sortpooling-k', type=float, default=0.6, help='the k use in sortpooling')
parser.add_argument('--lr', type=float, default=0.0001, help='the learning rate')
parser.add_argument('--n-paral', type=int, default=7, help='the number of parallel')
parser.add_argument('--epoch', type=int, default=150, help='epochs')
parser.add_argument('--xxx', type=int, default=0, help='set start method')
parser.add_argument('--pk', type=str, default='sort', help='pooling: sort or ')
parser.add_argument('--use-pagerank', default=False)
parser.add_argument('--m-size', type=int, default=0, help='set subgraph min size.')
parser.add_argument('--defense', type=int, default=1, help='1=use def; 0=no def.')
parser.add_argument('--gh', type=int, default=1, help='1=restructure S; 0=restructure H.')
parser.add_argument('--atk', type=int, default=1, help='0=random; 1=delete; 2=delete+add; 3=add; 4=DICE')
parser.add_argument('--atk-ratio', type=float, default=0.5, help='')


def get_current_memory_gb():
    pid = os.getpid()
    p = psutil.Process(pid)
    # print(p)进程号
    info = p.memory_full_info()
    # return info.uss / 1024 / 1024 / 1024
    print('当前进程的内存使用：%.4f GB' % (info.uss / 1024 / 1024 / 1024))


def label_statistics(real_labels):  # real_labels中的1,0指目标连边的同质异质性
    l = int(len(real_labels)/2)
    tongzhi_pos_link = 0
    yizhi_pos_link = 0
    tongzhi_neg_link = 0
    yizhi_neg_link = 0
    for i in range(l):
        if real_labels[i] == 1:
            tongzhi_pos_link += 1
        elif real_labels[i] == 0:
            yizhi_pos_link += 1
    for j in range(l, l*2):
        if real_labels[j] == 1:
            tongzhi_neg_link += 1
        elif real_labels[j] == 0:
            yizhi_neg_link += 1
    statistics = {"同质正连边": tongzhi_pos_link, "异质正连边": yizhi_pos_link, "同质负连边": tongzhi_neg_link, "异质负连边": yizhi_neg_link}

    return statistics


def atk_r(A, atk_link, labels):
    """随机边攻击"""
    count_ho_del = 0
    count_ho_add = 0
    count_he_del = 0
    count_he_add = 0
    for i in range(len(atk_link)):
        nei10 = neighbors([atk_link[i][0]], A)
        nei20 = neighbors(nei10, A) - nei10 - set([atk_link[i][0]])  # 获取到目标节点的纯二阶邻居
        nei11 = neighbors([atk_link[i][1]], A)
        nei21 = neighbors(nei11, A) - nei11 - set([atk_link[i][1]])
        nodes = [atk_link[i][0], atk_link[i][1]] + list(nei10) + list(nei20) + list(nei11) + list(nei21)

        flag0 = []
        flag1 = []
        if len(nodes) < 3:
            continue
        else:
            nei10 = list(nei10)
            nei11 = list(nei11)
            random.shuffle(nei10)
            random.shuffle(nei11)
            atk_n0 = nei10[:2]
            atk_n1 = nei11[:2]
            for j in range(len(atk_n0)):
                A[atk_link[i][0], atk_n0[j]] = A[atk_n0[j], atk_link[i][0]] = 0
            for k in range(len(atk_n1)):
                A[atk_link[i][1], atk_n1[k]] = A[atk_n1[k], atk_link[i][1]] = 0
            flag = 0
            while len(flag0) < 2 and flag < A.shape[0]:
                henode = random.randint(0, A.shape[0]-1)
                if henode not in nodes:  # 添加额外异质边
                    A[atk_link[i][0], henode] = A[henode, atk_link[i][0]] = 1
                    flag0.append(henode)
                    flag += 1
                    # print(flag, flush=True)
                else:
                    flag += 1
                    continue
            flag = 0
            while len(flag1) < 2 and flag < A.shape[0]:
                henode = random.randint(0, A.shape[0]-1)
                if henode not in nodes:  # 添加额外异质边
                    A[atk_link[i][1], henode] = A[henode, atk_link[i][1]] = 1
                    flag1.append(henode)
                    flag += 1
                    # print(flag, flush=True)
                else:
                    flag += 1
                    continue

    print('Attack-random done.')
    time.sleep(3)
    return A, count_ho_del, count_ho_add, count_he_del, count_he_add


def atk_d(A, atk_link, labels):
    """只删边的攻击"""
    degree = np.sum(A, axis=1)
    sortd = np.sort(degree, axis=0)
    index = int(len(degree)*0.4)
    de = int(sortd[index])
    count_ho_del = 0
    count_ho_add = 0
    count_he_del = 0
    count_he_add = 0
    for i in range(len(atk_link)):
        nei10 = neighbors([atk_link[i][0]], A)
        nei20 = neighbors(nei10, A) - nei10 -set([atk_link[i][0]])  # 获取到目标节点的纯二阶邻居
        nei11 = neighbors([atk_link[i][1]], A)
        nei21 = neighbors(nei11, A) - nei11 - set([atk_link[i][1]])

        nodes = [atk_link[i][0], atk_link[i][1]] + list(nei10) + list(nei20) + list(nei11) + list(nei21)
        # print('num of nodes: ', len(nodes))
        if len(nodes) < 3:
            continue
        else:
            nei10_ho = [node for node in nei10 if labels[atk_link[i][0]] == labels[node]]
            nei20_ho = [node for node in nei20 if labels[atk_link[i][0]] == labels[node]]
            nei11_ho = [node for node in nei11 if labels[atk_link[i][1]] == labels[node]]
            nei21_ho = [node for node in nei21 if labels[atk_link[i][1]] == labels[node]]
            ho_sum = len(nei10_ho)+len(nei20_ho)+len(nei11_ho)+len(nei21_ho)
            # print('hosum:', ho_sum)
            h2 = ho_sum/(len(nodes)-2)
            # print(h2)

            random.shuffle(nei10_ho)
            random.shuffle(nei11_ho)
            nei10_he = list(nei10 - set(nei10_ho))
            random.shuffle(nei10_he)
            nei11_he = list(nei11 - set(nei11_ho))
            random.shuffle(nei11_he)

            if h2 >= 0.5: # and len(nei10_ho) >= 2 and len(nei11_ho) >= 2:
                atk_n0 = nei10_ho[:2]
                atk_n1 = nei11_ho[:2]
                count_ho_del += len(atk_n0)
                count_ho_del += len(atk_n1)
                for j in range(len(atk_n0)):
                    A[atk_link[i][0], atk_n0[j]] = A[atk_n0[j], atk_link[i][0]] = 0
                for k in range(len(atk_n1)):
                    A[atk_link[i][1], atk_n1[k]] = A[atk_n1[k], atk_link[i][1]] = 0
            elif h2 < 0.5 and len(nei10) >= de and len(nei11) >= de:  #只有这种情况需要降低异质性
                atk_n0 = nei10_he[:2]
                atk_n1 = nei11_he[:2]
                count_he_del += len(atk_n0)
                count_he_del += len(atk_n1)
                for j in range(len(atk_n0)):
                    A[atk_link[i][0], atk_n0[j]] = A[atk_n0[j], atk_link[i][0]] = 0
                for k in range(len(atk_n1)):
                    A[atk_link[i][1], atk_n1[k]] = A[atk_n1[k], atk_link[i][1]] = 0
            elif h2 < 0.5 and (len(nei10) >= de and len(nei11) < de):
                atk_n0 = nei10_ho[:2]
                atk_n1 = nei11_ho[:2]
                count_ho_del += len(atk_n0)
                count_ho_del += len(atk_n1)
                for j in range(len(atk_n0)):
                    A[atk_link[i][0], atk_n0[j]] = A[atk_n0[j], atk_link[i][0]] = 0
                for k in range(len(atk_n1)):
                    A[atk_link[i][1], atk_n1[k]] = A[atk_n1[k], atk_link[i][1]] = 0
            elif h2 < 0.5 and (len(nei10) < de and len(nei11) >= de):
                atk_n0 = nei10_ho[:2]
                atk_n1 = nei11_ho[:2]
                count_ho_del += len(atk_n0)
                count_ho_del += len(atk_n1)
                for j in range(len(atk_n0)):
                    A[atk_link[i][0], atk_n0[j]] = A[atk_n0[j], atk_link[i][0]] = 0
                for k in range(len(atk_n1)):
                    A[atk_link[i][1], atk_n1[k]] = A[atk_n1[k], atk_link[i][1]] = 0
            else:
                continue
    print('Attack-del done.')
    time.sleep(3)
    return A, count_ho_del, count_ho_add, count_he_del, count_he_add


def atk_da(A, atk_link, labels):
    """既有删边又有加边的攻击"""
    # all_nodes = list(range(0, A.shape[0]))
    degree = np.sum(A, axis=1)
    sortd = np.sort(degree, axis=0)
    index = int(len(degree)*0.4)
    de = int(sortd[index])
    count_ho_del = 0
    count_ho_add = 0
    count_he_del = 0
    count_he_add = 0
    for i in range(len(atk_link)):
        nei10 = neighbors([atk_link[i][0]], A)
        nei20 = neighbors(nei10, A) - nei10 -set([atk_link[i][0]])  # 获取到目标节点的纯二阶邻居
        nei11 = neighbors([atk_link[i][1]], A)
        nei21 = neighbors(nei11, A) - nei11 - set([atk_link[i][1]])

        nodes = [atk_link[i][0], atk_link[i][1]] + list(nei10) + list(nei20) + list(nei11) + list(nei21)
        # print('num of nodes: ', len(nodes))
        if len(nodes) < 3:
            continue
        else:
            nei10_ho = [node for node in nei10 if labels[atk_link[i][0]] == labels[node]]
            nei20_ho = [node for node in nei20 if labels[atk_link[i][0]] == labels[node]]
            nei11_ho = [node for node in nei11 if labels[atk_link[i][1]] == labels[node]]
            nei21_ho = [node for node in nei21 if labels[atk_link[i][1]] == labels[node]]
            ho_sum = len(nei10_ho)+len(nei20_ho)+len(nei11_ho)+len(nei21_ho)
            # print('hosum:', ho_sum)
            h2 = ho_sum/(len(nodes)-2)
            # print(h2)

            random.shuffle(nei10_ho)
            random.shuffle(nei11_ho)
            nei10_he = list(nei10 - set(nei10_ho))
            random.shuffle(nei10_he)
            nei11_he = list(nei11 - set(nei11_ho))
            random.shuffle(nei11_he)

            flag0 = []
            flag1 = []
            if h2 >= 0.5: # and len(nei10_ho) >= 2 and len(nei11_ho) >= 2:
                atk_n0 = nei10_ho[:2]
                atk_n1 = nei11_ho[:2]
                count_ho_del += len(atk_n0)
                count_ho_del += len(atk_n1)
                for j in range(len(atk_n0)):
                    A[atk_link[i][0], atk_n0[j]] = A[atk_n0[j], atk_link[i][0]] = 0
                flag = 0
                while len(flag0) == 0 and flag < A.shape[0]:
                    henode = random.randint(0, A.shape[0]-1)
                    if labels[atk_link[i][0]] != labels[henode] and henode not in nodes:  # 添加额外异质边
                        A[atk_link[i][0], henode] = A[henode, atk_link[i][0]] = 1
                        flag0.append(henode)
                        count_he_add += 1
                        flag += 1
                        # print(flag, flush=True)
                    else:
                        flag += 1
                        continue
                for k in range(len(atk_n1)):
                    A[atk_link[i][1], atk_n1[k]] = A[atk_n1[k], atk_link[i][1]] = 0
                flag = 0
                while len(flag1) == 0 and flag < A.shape[0]:
                    henode = random.randint(0, A.shape[0]-1)
                    if labels[atk_link[i][1]] != labels[henode] and henode not in nodes:  # 添加额外异质边
                        A[atk_link[i][1], henode] = A[henode, atk_link[i][1]] = 1
                        flag1.append(henode)
                        count_he_add += 1
                        flag += 1
                        # print(flag, flush=True)
                    else:
                        flag += 1
                        continue
            elif h2 < 0.5 and len(nei10) >= de and len(nei11) >= de:  # 降低异质性
                atk_n0 = nei10_he[:2]
                atk_n1 = nei11_he[:2]
                count_he_del += len(atk_n0)
                count_he_del += len(atk_n1)
                for j in range(len(atk_n0)):
                    A[atk_link[i][0], atk_n0[j]] = A[atk_n0[j], atk_link[i][0]] = 0
                flag = 0
                while len(flag0) == 0 and flag < A.shape[0]:
                    honode = random.randint(0, A.shape[0]-1)
                    if labels[atk_link[i][0]] == labels[honode] and honode not in nodes:  # 添加额外同质边
                        A[atk_link[i][0], honode] = A[honode, atk_link[i][0]] = 1
                        flag0.append(honode)
                        count_ho_add += 1
                        flag += 1
                        # print(flag, flush=True)
                    else:
                        flag += 1
                        continue
                for k in range(len(atk_n1)):
                    A[atk_link[i][1], atk_n1[k]] = A[atk_n1[k], atk_link[i][1]] = 0
                flag = 0
                while len(flag1) == 0 and flag < A.shape[0]:
                    honode = random.randint(0, A.shape[0]-1)
                    if labels[atk_link[i][1]] == labels[honode] and honode not in nodes:  # 添加额外同质边
                        A[atk_link[i][1], honode] = A[honode, atk_link[i][1]] = 1
                        flag1.append(honode)
                        count_ho_add += 1
                        flag += 1
                        # print(flag, flush=True)
                    else:
                        flag += 1
                        continue
            elif h2 < 0.5 and (len(nei10) >= de and len(nei11) < de):
                atk_n0 = nei10_ho[:2]
                atk_n1 = nei11_ho[:2]
                count_ho_del += len(atk_n0)
                count_ho_del += len(atk_n1)
                for j in range(len(atk_n0)):
                    A[atk_link[i][0], atk_n0[j]] = A[atk_n0[j], atk_link[i][0]] = 0
                flag = 0
                while len(flag0) == 0 and flag < A.shape[0]:
                    henode = random.randint(0, A.shape[0]-1)
                    if labels[atk_link[i][0]] != labels[henode] and henode not in nodes:  # 添加额外异质边
                        A[atk_link[i][0], henode] = A[henode, atk_link[i][0]] = 1
                        flag0.append(henode)
                        count_he_add += 1
                        flag += 1
                        # print(flag, flush=True)
                    else:
                        flag += 1
                        continue
                for k in range(len(atk_n1)):
                    A[atk_link[i][1], atk_n1[k]] = A[atk_n1[k], atk_link[i][1]] = 0
                flag = 0
                while len(flag1) == 0 and flag < A.shape[0]:
                    henode = random.randint(0, A.shape[0]-1)
                    if labels[atk_link[i][1]] != labels[henode] and henode not in nodes:
                        A[atk_link[i][1], henode] = A[henode, atk_link[i][1]] = 1
                        flag1.append(henode)
                        count_he_add += 1
                        flag += 1
                        # print(flag, flush=True)
                    else:
                        flag += 1
                        continue
            elif h2 < 0.5 and (len(nei10) < de and len(nei11) >= de):
                atk_n0 = nei10_ho[:2]
                atk_n1 = nei11_ho[:2]
                count_ho_del += len(atk_n0)
                count_ho_del += len(atk_n1)
                for j in range(len(atk_n0)):
                    A[atk_link[i][0], atk_n0[j]] = A[atk_n0[j], atk_link[i][0]] = 0
                flag = 0
                while len(flag0) == 0 and flag < A.shape[0]:
                    henode = random.randint(0, A.shape[0]-1)
                    if labels[atk_link[i][0]] != labels[henode] and henode not in nodes:
                        A[atk_link[i][0], henode] = A[henode, atk_link[i][0]] = 1
                        flag0.append(henode)
                        count_he_add += 1
                        flag += 1
                        # print(flag, flush=True)
                    else:
                        flag += 1
                        continue
                for k in range(len(atk_n1)):
                    A[atk_link[i][1], atk_n1[k]] = A[atk_n1[k], atk_link[i][1]] = 0
                flag = 0
                while len(flag1) == 0 and flag < A.shape[0]:
                    henode = random.randint(0, A.shape[0]-1)
                    if labels[atk_link[i][1]] != labels[henode] and henode not in nodes:
                        A[atk_link[i][1], henode] = A[henode, atk_link[i][1]] = 1
                        flag1.append(henode)
                        count_he_add += 1
                        flag += 1
                        # print(flag, flush=True)
                    else:
                        flag += 1
                        continue
            else:
                continue
    print('Attack-del-add done.')
    time.sleep(3)
    return A, count_ho_del, count_ho_add, count_he_del, count_he_add


def atk_a(A, atk_link, labels):
    """只加边的攻击"""
    degree = np.sum(A, axis=1)
    sortd = np.sort(degree, axis=0)
    index = int(len(degree) * 0.4)
    de = int(sortd[index])
    count_ho_del = 0
    count_ho_add = 0
    count_he_del = 0
    count_he_add = 0
    for i in range(len(atk_link)):
        # 提二阶邻居
        nei10 = neighbors([atk_link[i][0]], A)
        nei20 = neighbors(nei10, A) - nei10 -set([atk_link[i][0]])  # 获取到目标节点的纯二阶邻居
        nei11 = neighbors([atk_link[i][1]], A)
        nei21 = neighbors(nei11, A) - nei11 - set([atk_link[i][1]])
        nodes = [atk_link[i][0], atk_link[i][1]] + list(nei10) + list(nei20) + list(nei11) + list(nei21)
        if len(nodes) < 3:
            continue
        else:
            nei10_ho = [node for node in nei10 if labels[atk_link[i][0]] == labels[node]]
            nei20_ho = [node for node in nei20 if labels[atk_link[i][0]] == labels[node]]
            nei11_ho = [node for node in nei11 if labels[atk_link[i][1]] == labels[node]]
            nei21_ho = [node for node in nei21 if labels[atk_link[i][1]] == labels[node]]
            ho_sum = len(nei10_ho) + len(nei20_ho) + len(nei11_ho) + len(nei21_ho)
            h2 = ho_sum / (len(nodes) - 2)

            flag0 = []
            flag1 = []
            if h2 >= 0.5:
                flag = 0
                while len(flag0) == 0 and flag < A.shape[0]:
                    henode = random.randint(0, A.shape[0]-1)
                    if labels[atk_link[i][0]] != labels[henode] and henode not in nodes:  # 添加额外异质边
                        A[atk_link[i][0], henode] = A[henode, atk_link[i][0]] = 1
                        flag0.append(henode)
                        count_he_add += 1
                        flag += 1
                        # print(flag, flush=True)
                    else:
                        flag += 1
                        continue
                flag = 0
                while len(flag1) == 0 and flag < A.shape[0]:
                    henode = random.randint(0, A.shape[0]-1)
                    if labels[atk_link[i][1]] != labels[henode] and henode not in nodes:  # 添加额外异质边
                        A[atk_link[i][1], henode] = A[henode, atk_link[i][1]] = 1
                        flag1.append(henode)
                        count_he_add += 1
                        flag += 1
                        # print(flag, flush=True)
                    else:
                        flag += 1
                        continue
            elif h2 < 0.5 and len(nei10) >= de and len(nei11) >= de:
                flag = 0
                while len(flag0) == 0 and flag < A.shape[0]:
                    honode = random.randint(0, A.shape[0]-1)
                    if labels[atk_link[i][0]] == labels[honode] and honode not in nodes:  # 添加额外同质边
                        A[atk_link[i][0], honode] = A[honode, atk_link[i][0]] = 1
                        flag0.append(honode)
                        count_ho_add += 1
                        flag += 1
                        # print(flag, flush=True)
                    else:
                        flag += 1
                        continue
                flag = 0
                while len(flag1) == 0 and flag < A.shape[0]:
                    honode = random.randint(0, A.shape[0]-1)
                    if labels[atk_link[i][1]] == labels[honode] and honode not in nodes:  # 添加额外同质边
                        A[atk_link[i][1], honode] = A[honode, atk_link[i][1]] = 1
                        flag1.append(honode)
                        count_ho_add += 1
                        flag += 1
                        # print(flag, flush=True)
                    else:
                        flag += 1
                        continue
            elif h2 < 0.5 and (len(nei10) >= de and len(nei11) < de):
                flag = 0
                while len(flag0) == 0 and flag < A.shape[0]:
                    henode = random.randint(0, A.shape[0]-1)
                    if labels[atk_link[i][0]] != labels[henode] and henode not in nodes:  # 添加额外异质边
                        A[atk_link[i][0], henode] = A[henode, atk_link[i][0]] = 1
                        flag0.append(henode)
                        count_he_add += 1
                        flag += 1
                        # print(flag, flush=True)
                    else:
                        flag += 1
                        continue
                flag = 0
                while len(flag1) == 0 and flag < A.shape[0]:
                    henode = random.randint(0, A.shape[0]-1)
                    if labels[atk_link[i][1]] != labels[henode] and henode not in nodes:  # 添加额外异质边
                        A[atk_link[i][1], henode] = A[henode, atk_link[i][1]] = 1
                        flag1.append(henode)
                        count_he_add += 1
                        flag += 1
                        # print(flag, flush=True)
                    else:
                        flag += 1
                        continue
            elif h2 < 0.5 and (len(nei10) < de and len(nei11) >= de):
                flag = 0
                while len(flag0) == 0 and flag < A.shape[0]:
                    henode = random.randint(0, A.shape[0]-1)
                    if labels[atk_link[i][0]] != labels[henode] and henode not in nodes:  # 添加额外异质边
                        A[atk_link[i][0], henode] = A[henode, atk_link[i][0]] = 1
                        flag0.append(henode)
                        count_he_add += 1
                        flag += 1
                        # print(flag, flush=True)
                    else:
                        flag += 1
                        continue
                flag = 0
                while len(flag1) == 0 and flag < A.shape[0]:
                    henode = random.randint(0, A.shape[0]-1)
                    if labels[atk_link[i][1]] != labels[henode] and henode not in nodes:  # 添加额外异质边
                        A[atk_link[i][1], henode] = A[henode, atk_link[i][1]] = 1
                        flag1.append(henode)
                        count_he_add += 1
                        flag += 1
                        # print(flag, flush=True)
                    else:
                        flag += 1
                        continue
            else:
                continue

    print('Attack-add done.')
    time.sleep(3)
    return A, count_ho_del, count_ho_add, count_he_del, count_he_add


def defen_atk(graph, node_features, gh):  # graph csc.matrix
    print('gh: ', gh)
    # graph = graph.A
    graph = np.triu(graph.A, k=0)  # 转为numpy格式, 取上半矩阵
    if gh >= 0.5:
        # print('nodes number of atk: ', sum(np.sum(graph, axis=1))/2)
        print('links before def:', sum(np.sum(graph, axis=1))/2)
        N, M = compute_NM(X=node_features, A=graph)
        lb2 = []
        for j in tqdm(range(N.shape[0])):
            lb = optimize_lbd2(N[j], M[j])
            lb2.append(lb)
        lb2 = np.array(lb2)
        S = op_S(lb2, N, M)
        S = A_final(S)
        S = np.triu(S, 1) + np.tril(S, -1)
        print('Matrix S is symmetric: ', np.allclose(S, np.transpose(S)))
        print('links after def:', sum(np.sum(S, axis=1))/2)
        # print('nodes number of def: ', len(nx_S.nodes))
        graph = S
    else:
        # print('nodes number of atk: ', graph_list[i].num_nodes)
        print('links before def:', sum(np.sum(graph, axis=1))/2)
        H = H_final(graph, torch.tensor(node_features, dtype=torch.float))
        H = np.triu(H, 1) + np.tril(H, -1)
        print('Matrix H is symmetric: ', np.allclose(H, np.transpose(H)))
        print('links after def:', sum(np.sum(H, axis=1))/2)
        # print('nodes number of def: ', len(nx_H.nodes))
        graph = H

    return graph


def resample_link(A, test_pos, max_train_num=None):
    A_triu = ssp.triu(A, k=1)
    Arow, Acol, _ = ssp.find(A_triu)
    train_pos = (Arow[:], Acol[:])
    if max_train_num is not None and train_pos is not None:
        perm = np.random.permutation(len(train_pos[0]))[:max_train_num]  # permutation自带随机排序功能
        train_pos = (train_pos[0][perm], train_pos[1][perm])
    train_num = len(train_pos[0])

    test_set = []
    for i in range(len(test_pos[0])):
        test_set.append([test_pos[0][i], test_pos[1][i]])

    neg = ([], [])
    n = A.shape[0]
    while len(neg[0]) < train_num + len(test_pos[0]):
        i, j = random.randint(0, n - 1), random.randint(0, n - 1)
        if i < j and A[i, j] == 0 and [i, j] not in test_set:
            neg[0].append(i)
            neg[1].append(j)
        else:
            continue
    train_neg = (neg[0][:train_num], neg[1][:train_num])
    test_neg = (neg[0][train_num:], neg[1][train_num:])

    return train_pos, train_neg, test_neg


args = parser.parse_args()
if args.xxx == 1:
    mp.set_start_method('forkserver', force=True)
if __name__ == '__main__':
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print(args)

    if args.hop != 'auto':
        args.hop = int(args.hop)

    '''Prepare data'''
    args.file_dir = os.path.dirname(os.path.realpath('__file__'))

    # check whether train and test links are provided
    train_pos, test_pos = None, None
    if args.train_name is not None:
        args.train_dir = os.path.join(args.file_dir, 'data/{}'.format(args.train_name))
        train_idx = np.loadtxt(args.train_dir, dtype=int)
        train_pos = (train_idx[:, 0], train_idx[:, 1])
    if args.test_name is not None:
        args.test_dir = os.path.join(args.file_dir, 'data/{}'.format(args.test_name))
        test_idx = np.loadtxt(args.test_dir, dtype=int)
        test_pos = (test_idx[:, 0], test_idx[:, 1])

    # build observed network
    if args.d_name is not None:  # use .mat network
        if args.d_name=='syn-cora':
            print("Using synthetic cora...")
            '''
            hname = 'h' + str(args.hh) + '0-r2'
            dataset = CustomDataset(root="syn-cora", name=hname, setting="gcn", seed=15)
            adj = dataset.adj
            attributes = dataset.features
            node_label = dataset.labels
            net = adj.tocsc(adj)
            # print(type(net))
            '''
        elif args.d_name=='cora' or args.d_name=='citeseer' or args.d_name=='pubmed':
            def setlabels(label_list):
                # 从大到小排序，L中第一个元组记录着值为1的元素的下标与该元素
                L = [sorted(enumerate(label_list[i]), key=lambda x: x[1], reverse=True)[0][0] for i in range(len(label_list))]
                L = np.array(L)  # 转成所需的形式
                return L
            d_name = 'ind.' + args.d_name
            dataset = PlanetoidData(dataset_str=d_name, dataset_path='data/inddata')
            adj, features, _, _, _, _, _, _, _, _, labels = dataset.load_data(dataset_str=dataset.dataset_str)
            # print(type(adj)) # 'scipy.sparse.csr.csr_matrix'
            # print(type(features)) # 'scipy.sparse.lil.lil_matrix'
            attributes = features.toarray()  # 'numpy.ndarrary'
            single_labels = setlabels(labels)  # 记录每个节点label的一维列表，label范围0~6
            print("nodes: ", len(single_labels))
            # print(adj.nnz)
            adj = adj.A
            for i in range(len(single_labels)):  #去自环
                adj[i][i] = 0
            adj = ssp.csr_matrix(adj)
            net = adj.tocsc()
            # print(net.nnz)
            # time.sleep(100)
        elif args.d_name=='actor' or args.d_name=='chameleon'or args.d_name=='cornell' or args.d_name=='texas'\
            or args.d_name=='wisconsin' or args.d_name=='chameleon_filtered' or args.d_name=='squirrel_filtered':
            # actor:node7600  chameleon: node2277,link31371
            # chameleon_filtered:node864,link7754  squirrel_filtered:node2205,link46557直接跑跑不动
            d_name = args.d_name + '.npz'
            G = load_npzdata(d_name)
            adj = G['adj']
            attributes = G['features']
            single_labels = G['labels']
            print("nodes: ", len(single_labels))
            adj = adj.A
            for i in range(len(single_labels)):  # 去自环
                adj[i][i] = 0
            adj = ssp.csr_matrix(adj)
            net = adj.tocsc()
        elif args.d_name=='ENGB' or args.d_name=='ES' or args.d_name=='PTBR' or args.d_name=='RU' or args.d_name=='TW':
            twitch_dataset = twitch.load_twitch_dataset(args.d_name)
            adj = twitch_dataset.graph['adj']
            attributes = twitch_dataset.graph['node_feat'].numpy()
            single_labels = twitch_dataset.label.numpy()
            print("nodes: ", len(single_labels))
            net = adj.tocsc(adj)
        elif args.d_name=='Reed98':
            facebook_dataset = twitch.load_fb100_dataset(args.d_name)
            net = facebook_dataset.graph['adj']
            attributes = facebook_dataset.graph['node_feat'].numpy()
            single_labels = facebook_dataset.label.numpy()
        else:
            args.d_dir = os.path.join(args.file_dir, 'data/{}.mat'.format(args.d_name))
            data = sio.loadmat(args.d_dir)
            net = data['net']
            # print(type(data))
            print(type(data['net']))
            if 'group' in data:
                # load node attributes (here a.k.a. node classes)
                attributes = data['group'].toarray().astype('float32')
                # print(type(attributes))
                # print('feature')
            else:
                attributes = None
        # check whether net is symmetric (for small nets only)
        if False:
            net_ = net.toarray()
            assert(np.allclose(net_, net_.T, atol=1e-8))
    else:  # build network from train links
        assert (args.train_name is not None), "must provide train links if not using .mat"
        if args.train_name.endswith('_train.txt'):
            args.d_name = args.train_name[:-10]
        else:
            args.d_name = args.train_name.split('.')[0]
        max_idx = np.max(train_idx)
        if args.test_name is not None:
            max_idx = max(max_idx, np.max(test_idx))
        net = ssp.csc_matrix((np.ones(len(train_idx)), (train_idx[:, 0], train_idx[:, 1])), shape=(max_idx+1, max_idx+1))
        net[train_idx[:, 1], train_idx[:, 0]] = 1  # add symmetric edges
        net[np.arange(max_idx+1), np.arange(max_idx+1)] = 0  # remove self-loops

    # sample train and test links
    if args.train_name is None and args.test_name is None:
        if args.d_name == 'syn-cora':
            train_pos, train_neg, test_pos, test_neg = sample_link(net, args.hh, node_label)
            print('train links: %d, test links: %d' % (len(train_pos), len(test_pos)))
        else:  # sample both positive and negative train/test links from net
            train_pos, train_neg, test_pos, test_neg = sample_neg(net, args.test_ratio, max_train_num=args.max_train,
                                                                  max_test_num=args.max_test)
            print('Train links: %d, test links: %d' % (len(train_pos[0]), len(test_pos[0])))
            time.sleep(2)
    else:
        # use provided train/test positive links, sample negative from net
        train_pos, train_neg, test_pos, test_neg = sample_neg(net, train_pos=train_pos, test_pos=test_pos,
                                            max_train_num=args.max_train, max_test_num=args.max_test,
                                            all_unknown_as_negative=args.all_unknown_as_negative)

    # print(net.nnz)
    gh = graph_link_homo(net, single_labels)
    # print(gh)
    # net = defen_atk(net, attributes, gh=1)
    # A2 = net.dot(net)
    # A2 = A2.A
    # A2 = np.triu(A2, 1) + np.tril(A2, -1)
    # index = np.where(A2 == 2)
    # for i in range(len(index[0])):
    #     A2[index[0][i]][index[1][i]] = A2[index[1][i]][index[0][i]] = 1
    # A2 = ssp.csc_matrix(A2)
    # print(A2.nnz)
    # gh2 = graph_link_homo(A2, single_labels)
    # print(gh2)
    # time.sleep(100)
    '''Train and apply classifier'''
    A = net.copy()  # the observed network
    A[test_pos[0], test_pos[1]] = 0  # mask test links
    A[test_pos[1], test_pos[0]] = 0  # mask test links
    A.eliminate_zeros()  # make sure the links are masked when using the sparse matrix in scipy-1.3.x
    # print(type(A))  # <class 'scipy.sparse.csc.csc_matrix'>

    AA = A.A
    if not np.allclose(AA, np.transpose(AA)):
        print("the net is not symmetric, so...")
        AA = AA + np.transpose(AA)  # 确保成为实对称矩阵
        A = ssp.csr_matrix(AA).tocsc()
        # print('Matrix AA is symmetric: ', np.allclose(AA, np.transpose(AA)))
        # time.sleep(100)

    linknum = int(sum(np.sum(A, axis=1)) / 2)
    print("link number before atk: ", linknum)
    with open("results_txt/" + args.d_name + "_atk_result_k" + str(int(args.sortpooling_k * 10)) + ".txt", "a+") as ff:
        ff.write("ori graph homo ratio:" + str(gh) + '\n')
        ff.write("link number before atk:" + str(linknum) + '\n')
        ff.write("attributes:" + str(args.use_attribute) + '\n')

    node_information = None
    if args.use_embedding:
        embeddings = generate_node2vec_embeddings(A, 128, True, train_neg)
        node_information = embeddings
    if args.use_attribute and attributes is not None:
        if node_information is not None:
            node_information = np.concatenate([node_information, attributes], axis=1)
        else:
            print('Use attributes.')
            node_information = attributes

    # A_atk = A.copy()
    if args.only_predict:  # no need to use negatives
        # test_pos is a name only, we don't actually know their labels
        _, test_graphs, real_labels, max_n_label = links2subgraphs(A, None, None, test_pos, None, single_labels, args.hop,
                                    args.max_nodes_per_hop, node_information, args.no_parallel)
        print('# test: %d' % (len(test_graphs)))
    else:
        if args.defense == 0 and args.atk != 4:
            train_graphs, test_graphs1, max_n_label1 = links2subgraphs(A, train_pos, train_neg, [], test_neg, single_labels,
                                                            args.n_paral, args.hop, args.m_size,
                                                            args.max_nodes_per_hop, node_information,
                                                            args.no_parallel, args.use_pagerank)

            ##此处选取攻击对象可以有不同方案，主要看是随机选取50%，还是根据子图的大小选25%-75%这段，或者专门选择子图同质性高的一批
            atk_link_idx = random.sample(range(len(test_pos[0])), int(len(test_pos[0])*args.atk_ratio))
            atk_link = np.zeros([len(atk_link_idx), 2], dtype=int)
            for ii in range(len(atk_link_idx)):
                atk_link[ii][0] = test_pos[0][atk_link_idx[ii]]
                atk_link[ii][1] = test_pos[1][atk_link_idx[ii]]
            if args.atk == 0:
                A, n_ho_del, n_ho_add, n_he_del, n_he_add = atk_r(A, atk_link, single_labels)
            elif args.atk == 1:
                A, n_ho_del, n_ho_add, n_he_del, n_he_add = atk_d(A, atk_link, single_labels)
            elif args.atk == 2:
                A, n_ho_del, n_ho_add, n_he_del, n_he_add = atk_da(A, atk_link, single_labels)
            elif args.atk == 3:
                A, n_ho_del, n_ho_add, n_he_del, n_he_add = atk_a(A, atk_link, single_labels)

            linknum = int(sum(np.sum(A, axis=1)) / 2)
            print("link number after atk: ", linknum)
            with open("results_txt/" + args.d_name + "_atk_result_k" + str(int(args.sortpooling_k*10)) + ".txt",
                      "a+") as ff:
                ff.write("num of ho-del:" + str(n_ho_del) + '\n')
                ff.write("num of ho-add:" + str(n_ho_add) + '\n')
                ff.write("num of he-del:" + str(n_he_del) + '\n')
                ff.write("num of he-add:" + str(n_he_add) + '\n')
                ff.write("atk graph homo ratio:" + str(graph_link_homo(A, single_labels)) + '\n')
                ff.write("link number after atk:" + str(linknum) + '\n')

            _, test_graphs2, max_n_label2 = links2subgraphs(A, [], [], test_pos, [], single_labels,
                                                            args.n_paral, args.hop, args.m_size,
                                                            args.max_nodes_per_hop, node_information,
                                                            args.no_parallel, args.use_pagerank)
            test_graphs = test_graphs2 + test_graphs1
            max_n_label = max(max_n_label1, max_n_label2)
        elif args.defense == 0 and args.atk == 4:
            model = DICE()
            model.attack(A, single_labels, n_perturbations=int(len(test_pos[0]) * args.atk_ratio))
            A = model.modified_adj

            linknum = int(sum(np.sum(A, axis=1)) / 2)
            print("link number after atk: ", linknum)
            # number of pertubations: cora263、citeseer227、pubmed2216、cornell13、texas13、wisconsin22、actor1332
            with open("results_txt/" + args.d_name + "_atk_result_k" + str(int(args.sortpooling_k * 10)) + ".txt",
                      "a+") as ff:
                ff.write("atk-DICE" + '\n')
                ff.write("atk graph homo ratio:" + str(graph_link_homo(A, single_labels)) + '\n')
                ff.write("link number after atk:" + str(linknum) + '\n')

            train_graphs, test_graphs, max_n_label = links2subgraphs(A, train_pos, train_neg, test_pos, test_neg,
                                                            single_labels, args.n_paral, args.hop, args.m_size,
                                                            args.max_nodes_per_hop, node_information,
                                                            args.no_parallel, args.use_pagerank)

        if args.defense == 1:
            atk_link_idx = random.sample(range(len(test_pos[0])), int(len(test_pos[0]) * args.atk_ratio))
            atk_link = np.zeros([len(atk_link_idx), 2], dtype=int)
            for ii in range(len(atk_link_idx)):
                atk_link[ii][0] = test_pos[0][atk_link_idx[ii]]
                atk_link[ii][1] = test_pos[1][atk_link_idx[ii]]
            if args.atk == 0:
                A, n_ho_del, n_ho_add, n_he_del, n_he_add = atk_r(A, atk_link, single_labels)
            if args.atk == 1:
                A, n_ho_del, n_ho_add, n_he_del, n_he_add = atk_d(A, atk_link, single_labels)
            if args.atk == 2:
                A, n_ho_del, n_ho_add, n_he_del, n_he_add = atk_da(A, atk_link, single_labels)
            if args.atk == 3:
                A, n_ho_del, n_ho_add, n_he_del, n_he_add = atk_a(A, atk_link, single_labels)
            if args.atk == 4:
                model = DICE()
                model.attack(A, single_labels, n_perturbations=int(len(test_pos[0]) * args.atk_ratio)) #2
                A = model.modified_adj
                n_ho_del = n_ho_add = n_he_del = n_he_add = 'DICE'
            if args.atk != 9:
                linknum = int(sum(np.sum(A, axis=1)) / 2)
                print("link number after atk: ", linknum)
                with open("results_txt/" + args.d_name + "_atk_result_k" + str(int(args.sortpooling_k * 10)) + ".txt",
                          "a+") as ff:
                    ff.write("num of ho-del:" + str(n_ho_del) + '\n')
                    ff.write("num of ho-add:" + str(n_ho_add) + '\n')
                    ff.write("num of he-del:" + str(n_he_del) + '\n')
                    ff.write("num of he-add:" + str(n_he_add) + '\n')
                    ff.write("atk graph homo ratio:" + str(graph_link_homo(A, single_labels)) + '\n')
                    ff.write("link number after atk:" + str(linknum) + '\n')

            A = defen_atk(A, attributes, args.gh)
            for n in range(len(test_pos[0])):  # 重新隐去测试边
                A[test_pos[0][n]][test_pos[1][n]] = A[test_pos[1][n]][test_pos[0][n]] = 0
            A = np.array(AA, dtype=int) | np.array(A, dtype=int)
            print('Matrix AA is symmetric: ', np.allclose(A, np.transpose(A)))
            A = ssp.csr_matrix(A)
            A = A.tocsc()

            train_pos, train_neg, test_neg = resample_link(A, test_pos, max_train_num=args.max_train)
            print('New train links: ', len(train_pos[0]))
            # print('New train links neg: ', len(train_neg[0]))
            # print('New test neg: ', len(test_neg[0]))
            # time.sleep(100)

            with open("results_txt/" + args.d_name + "_atk_result_k" + str(int(args.sortpooling_k*10)) + ".txt", "a+") as ff:
                ff.write("def graph homo ratio:" + str(graph_link_homo(A, single_labels)) + '\n')
                ff.write("link number after def:" + str(A.nnz/2) + '\n')
            # time.sleep(100)

            train_graphs, test_graphs, max_n_label = links2subgraphs(A, train_pos, train_neg, test_pos, test_neg, single_labels,
                                                       args.n_paral, args.hop, args.m_size,
                                                       args.max_nodes_per_hop, node_information,
                                                       args.no_parallel, args.use_pagerank, args.defense)

        print('# train G: %d, # test G: %d' % (len(train_graphs), len(test_graphs)))
        # time.sleep(100)

    real_labels = [test_g.link_label for test_g in test_graphs]
    statistics_result = label_statistics(real_labels)

    with open("results_txt/" + args.d_name + "_atk_result_k" + str(int(args.sortpooling_k*10)) + ".txt", "a+") as ff:
        ff.write("Train links:" + str(len(train_pos[0])) + " test links:" + str(len(test_pos[0])) + '\n')
        ff.write("min subgraph size:" + str(args.m_size) + '\n')
        ff.write("train G:" + str(len(train_graphs)) + " test G:" + str(len(test_graphs)) + '\n')
        ff.write("batch size:" + str(args.batch_size) + " learning rate:" + str(args.lr) + " pool:" + args.pk + '\n')
        ff.write("Use personalization pr: " + str(args.use_pagerank) + '\n')
        ff.write(str(statistics_result) + '\n')

    # DGCNN configurations
    if args.only_predict:
        with open('data/{}_hyper.pkl'.format(args.d_name), 'rb') as hyperparameters_name:
            saved_cmd_args = pickle.load(hyperparameters_name)
        for key, value in vars(saved_cmd_args).items(): # replace with saved cmd_args
            vars(cmd_args)[key] = value
        classifier = Classifier()
        if cmd_args.mode == 'gpu':
            classifier = classifier.cuda()
        model_name = 'data/{}_model.pth'.format(args.d_name)
        classifier.load_state_dict(torch.load(model_name))
        classifier.eval()
        predictions = []
        batch_graph = []
        for i, graph in enumerate(test_graphs):
            batch_graph.append(graph)
            if len(batch_graph) == cmd_args.batch_size or i == (len(test_graphs)-1):
                predictions.append(classifier(batch_graph)[0][:, 1].exp().cpu().detach())
                batch_graph = []
        predictions = torch.cat(predictions, 0).unsqueeze(1).numpy()
        test_idx_and_pred = np.concatenate([test_idx, predictions], 1)
        pred_name = 'data/' + args.test_name.split('.')[0] + '_pred.txt'
        np.savetxt(pred_name, test_idx_and_pred, fmt=['%d', '%d', '%1.2f'])
        print('Predictions for {} are saved in {}'.format(args.test_name, pred_name))
        exit()

    cmd_args.mode = 'gpu' if args.cuda else 'cpu'
    cmd_args.gm = 'DGCNN'
    cmd_args.pool_kind = args.pk
    cmd_args.feat_dim = max_n_label + 1
    cmd_args.num_class = 2
    cmd_args.num_epochs = args.epoch
    cmd_args.latent_dim = [32, 32, 32, 1]
    cmd_args.sortpooling_k = args.sortpooling_k
    cmd_args.out_dim = 0
    cmd_args.hidden = 128
    cmd_args.learning_rate = args.lr
    cmd_args.dropout = True
    cmd_args.printAUC = True
    # cmd_args.attr_dim = 0
    if node_information is not None:
        cmd_args.attr_dim = node_information.shape[1]  # Classifier初始化的时候里面有attr_dim参数,所以是有影响的
        print("The dim of attributes is:", cmd_args.attr_dim)
    elif node_information is None:
        cmd_args.attr_dim = 0

    if cmd_args.sortpooling_k <= 1:
        num_nodes_list = sorted([g.num_nodes for g in train_graphs + test_graphs])  # 所有子图含有的节点数量的排序列表
        k_ = int(math.ceil(cmd_args.sortpooling_k * len(num_nodes_list))) - 1  # 通过s_k*所有子图的总数=下标k_
        cmd_args.sortpooling_k = max(1, num_nodes_list[k_])  # 真正的sortpooling_k要么是10，要么是num_nodes_list中第k_个值
        # print('k used in SortPooling is: ' + str(cmd_args.sortpooling_k))
        with open("results_txt/" + args.d_name + "_atk_result_k" + str(int(args.sortpooling_k*10)) + ".txt", "a+") as ff:
            ff.write('k used in SortPooling is: ' + str(cmd_args.sortpooling_k) + '\n')

    classifier = Classifier()
    if cmd_args.mode == 'gpu':
        classifier = classifier.cuda()
    # print('num_node_feats', classifier.gnn.num_node_feats)
    optimizer = optim.Adam(classifier.parameters(), lr=cmd_args.learning_rate)

    random.shuffle(train_graphs)  # 随机打乱
    val_num = int(0.1 * len(train_graphs))
    val_graphs = train_graphs[:val_num]
    train_graphs = train_graphs[val_num:]

    train_idxes = list(range(len(train_graphs)))
    best_loss = None
    best_epoch = None
    for epoch in range(cmd_args.num_epochs):
        # get_current_memory_gb()
        random.shuffle(train_idxes)  # 下标被打乱,这一步很重要必须有,关系到loop_dataset里的total_iters问题
        classifier.train()
        avg_loss, _ = loop_dataset(train_graphs, classifier, train_idxes, optimizer=optimizer, bsize=args.batch_size)
        if not cmd_args.printAUC:
            avg_loss[2] = 0.0
        print('\033[92maverage training of epoch %d: loss %.5f acc %.5f auc %.5f ap %.5f\033[0m'
              % (epoch, avg_loss[0], avg_loss[1], avg_loss[2], avg_loss[3]))

        classifier.eval()
        val_loss, _ = loop_dataset(val_graphs, classifier, list(range(len(val_graphs))))
        if not cmd_args.printAUC:
            val_loss[2] = 0.0
        print('\033[93maverage validation of epoch %d: loss %.5f acc %.5f auc %.5f ap %.5f\033[0m'
              % (epoch, val_loss[0], val_loss[1], val_loss[2], val_loss[3]))
        if best_loss is None:
            best_loss = val_loss
        if val_loss[0] <= best_loss[0]:  # 将当前与之前的最佳loss做比较
            best_loss = val_loss
            best_epoch = epoch
            # 测试集并未被打乱,前一半正样本，后一半负
            test_loss, total_pre_result = loop_dataset(test_graphs, classifier, list(range(len(test_graphs))))

            count_result = compare_result(real_labels, total_pre_result)
            if not cmd_args.printAUC:
                test_loss[2] = 0.0
            print('\033[94maverage test of epoch %d: loss %.5f acc %.5f auc %.5f ap %.5f\033[0m'
                  % (epoch, test_loss[0], test_loss[1], test_loss[2], test_loss[3]))
            # print(statistics_result)
            print(count_result)  # 把比较结果存进txt会更方便
            with open("results_txt/" + args.d_name + "_atk_result_k" + str(int(args.sortpooling_k*10)) + ".txt", "a+") as ff:
                ff.write(str(epoch) + ': ')
                ff.write(str(count_result))
                ff.write('test of epoch %d: loss %.5f acc %.5f auc %.5f ap %.5f'
                         % (epoch, test_loss[0], test_loss[1], test_loss[2], test_loss[3]))
                ff.write('\n')

    with open("results_txt/" + args.d_name + "_atk_result_k" + str(int(args.sortpooling_k*10)) + ".txt", "a+") as ff:
        ff.write('Final test performance: epoch %d: loss %.5f acc %.5f auc %.5f ap %.5f'
                 % (best_epoch, test_loss[0], test_loss[1], test_loss[2], test_loss[3]))
        ff.write('\n')
    print('\033[95mFinal test performance: epoch %d: loss %.5f acc %.5f auc %.5f ap %.5f\033[0m'
          % (best_epoch, test_loss[0], test_loss[1], test_loss[2], test_loss[3]) + '\n')

    if args.save_model:
        model_name = 'model/{}_model.pth'.format(args.d_name)
        print('Saving final model states to {}...'.format(model_name))
        torch.save(classifier.state_dict(), model_name)
        hyper_name = 'data/{}_hyper.pkl'.format(args.d_name)
        with open(hyper_name, 'wb') as hyperparameters_file:
            pickle.dump(cmd_args, hyperparameters_file)
            print('Saving hyperparameters to {}...'.format(hyper_name))
