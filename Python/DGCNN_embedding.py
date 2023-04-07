from __future__ import print_function
import os
import sys
import time
import numpy as np
import torch
import random
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import pdb
from torch_geometric.nn import GCNConv
from torch_geometric.nn.pool.topk_pool import topk

sys.path.append('%s/lib' % os.path.dirname(os.path.realpath(__file__)))
from lib.gnn_lib import GNNLIB
from lib.pytorch_util import weights_init, gnn_spmm


class DGCNN(nn.Module):
    def __init__(self, output_dim, num_node_feats, num_edge_feats, latent_dim=[32, 32, 32, 1], k=30,
                 conv1d_channels=[16, 32], conv1d_kws=[0, 5], conv1d_activation='ReLU'):
        print('Initializing DGCNN')
        super(DGCNN, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.num_node_feats = num_node_feats
        self.num_edge_feats = num_edge_feats
        self.k = k
        self.total_latent_dim = sum(latent_dim)
        conv1d_kws[0] = self.total_latent_dim

        self.conv_params = nn.ModuleList()
        self.conv_params.append(nn.Linear(num_node_feats + num_edge_feats, latent_dim[0]))
        for i in range(1, len(latent_dim)):
            self.conv_params.append(nn.Linear(latent_dim[i-1], latent_dim[i]))

        self.conv1d_params1 = nn.Conv1d(1, conv1d_channels[0], conv1d_kws[0], conv1d_kws[0])
        self.maxpool1d = nn.MaxPool1d(2, 2)
        self.conv1d_params2 = nn.Conv1d(conv1d_channels[0], conv1d_channels[1], conv1d_kws[1], 1)

        dense_dim = int((k - 2) / 2 + 1)
        self.dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]

        # if num_edge_feats > 0:
        #    self.w_e2l = nn.Linear(num_edge_feats, num_node_feats)
        if output_dim > 0:
            self.out_params = nn.Linear(self.dense_dim, output_dim)  # 设置网络中的全连接层

        self.conv1d_activation = eval('nn.{}()'.format(conv1d_activation))  # 执行一个字符串表达式，并返回表达式的值

        weights_init(self)

        self.poolConv = GCNConv(self.total_latent_dim, 1)

    def forward(self, graph_list, node_feat, edge_feat):#, batch):
        graph_sizes = [graph_list[i].num_nodes for i in range(len(graph_list))]
        num_edges_set = [graph_list[j].num_edges for j in range(len(graph_list))]
        # 准备GCNConv计算score所需的edge_index
        '''
        edge1 = graph_list[0].edge_pairs[0::2]
        edge2 = graph_list[0].edge_pairs[1::2]
        for m in range(1, len(graph_list)):
            edge1 = np.append(edge1, graph_list[m].edge_pairs[0::2]+sum(graph_sizes[:m]))
            edge2 = np.append(edge2, graph_list[m].edge_pairs[0::2]+sum(graph_sizes[:m]))
        edge1 = torch.LongTensor(np.array([edge1]))
        edge2 = torch.LongTensor(np.array([edge2]))
        edge_index = torch.cat([edge1, edge2], 0)
        '''
        edge1 = graph_list[0].edge_pairs[0::2]
        edge2 = graph_list[0].edge_pairs[1::2]
        # time.sleep(100)
        for m in range(1, len(graph_list)):
            edge1 = np.append(edge1, graph_list[m].edge_pairs[0::2])
            edge2 = np.append(edge2, graph_list[m].edge_pairs[1::2])
        edge1 = torch.LongTensor(np.array([edge1]))
        edge2 = torch.LongTensor(np.array([edge2]))
        edge_index = torch.cat([edge1, edge2], 0)  # 每个子图含有graph_list[m].num_edges条连边

        # degs是degrees
        node_degs = [torch.Tensor(graph_list[i].degs) + 1 for i in range(len(graph_list))]
        node_degs = torch.cat(node_degs).unsqueeze(1)

        n2n_sp, e2n_sp, subg_sp = GNNLIB.PrepareSparseMatrices(graph_list)

        if torch.cuda.is_available() and isinstance(node_feat, torch.cuda.FloatTensor):
            n2n_sp = n2n_sp.cuda()
            e2n_sp = e2n_sp.cuda()
            subg_sp = subg_sp.cuda()
            node_degs = node_degs.cuda()
            edge_index = edge_index.cuda()
        node_feat = Variable(node_feat)
        if edge_feat is not None:
            edge_feat = Variable(edge_feat)
            if torch.cuda.is_available() and isinstance(node_feat, torch.cuda.FloatTensor):
                edge_feat = edge_feat.cuda()
        n2n_sp = Variable(n2n_sp)
        e2n_sp = Variable(e2n_sp)
        subg_sp = Variable(subg_sp)
        node_degs = Variable(node_degs)
        edge_index = Variable(edge_index)

        h = self.sortpooling_embedding(node_feat, edge_feat, n2n_sp, e2n_sp, subg_sp, graph_sizes, node_degs)
        # h = self.attpooling_embedding(node_feat, edge_feat, edge_index, num_edges_set, n2n_sp, e2n_sp, subg_sp, graph_sizes, node_degs)

        return h

    def sortpooling_embedding(self, node_feat, edge_feat, n2n_sp, e2n_sp, subg_sp, graph_sizes, node_degs):
        """if exists edge feature, concatenate to node feature vector"""
        if edge_feat is not None:
            print("edge_feat is none.")
            # input_edge_linear = self.w_e2l(edge_feat)
            input_edge_linear = edge_feat
            e2npool_input = gnn_spmm(e2n_sp, input_edge_linear)
            node_feat = torch.cat([node_feat, e2npool_input], 1)
        # print("nodefeat:", node_feat.size())  # torch.Size([xxxx, 1569])

        """graph convolution layers"""
        lv = 0
        cur_message_layer = node_feat
        cat_message_layers = []
        while lv < len(self.latent_dim):
            # gnn_spmm就是矩阵乘法
            n2npool = gnn_spmm(n2n_sp, cur_message_layer) + cur_message_layer  # Y = (A + I) * X = AX+X
            node_linear = self.conv_params[lv](n2npool)  # Y = Y * W
            normalized_linear = node_linear.div(node_degs)  # Y = D^-1 * Y
            cur_message_layer = torch.tanh(normalized_linear)
            cat_message_layers.append(cur_message_layer)
            lv += 1

        cur_message_layer = torch.cat(cat_message_layers, 1)  # 将layers中的张量按第1维度进行拼接,如两个size为[2,3]的张量拼成一个[2,6]
        # print(cur_message_layer.size())  # torch.Size([xxxx, 97]) xxxx与前面的node_feat一致

        """sortpooling layer"""
        sort_channel = cur_message_layer[:, -1]  # 将最后一列取出, 要比较的是最后一列的数值
        batch_sortpooling_graphs = torch.zeros(len(graph_sizes), self.k, self.total_latent_dim)  # graph_sizes是每个子图大小,长度是一个batch
        # print(batch_sortpooling_graphs.size())
        if torch.cuda.is_available() and isinstance(node_feat.data, torch.cuda.FloatTensor):
            batch_sortpooling_graphs = batch_sortpooling_graphs.cuda()
        batch_sortpooling_graphs = Variable(batch_sortpooling_graphs)  # 将Tensor转换为Variable之后，可以装载梯度信息

        accum_count = 0
        for i in range(subg_sp.size()[0]):  # subg_sp,torch.Size([128, 10844])
            to_sort = sort_channel[accum_count: accum_count + graph_sizes[i]]
            k = self.k if self.k <= graph_sizes[i] else graph_sizes[i]
            _, topk_indices = to_sort.topk(k)  # torch.Tensor
            # print(topk_indices)
            '''tensor.int64([ 3, 64, 28, 35,  7, ......13, 65, 78], device='cuda:0')'''
            topk_indices += accum_count
            # print(topk_indices)
            sortpooling_graph = cur_message_layer.index_select(0, topk_indices)  # _select最后生成的张量总维度不变,0代表选择维度
            # 如果图的大小<k, 就会把缺的部分用0补齐
            if k < self.k:
                to_pad = torch.zeros(self.k-k, self.total_latent_dim)
                if torch.cuda.is_available() and isinstance(node_feat.data, torch.cuda.FloatTensor):
                    to_pad = to_pad.cuda()
                to_pad = Variable(to_pad)
                sortpooling_graph = torch.cat((sortpooling_graph, to_pad), 0)
            batch_sortpooling_graphs[i] = sortpooling_graph  # 存放每一个池化后的子图
            accum_count += graph_sizes[i]

        """traditional 1d convlution and dense layers"""
        to_conv1d = batch_sortpooling_graphs.view((-1, 1, self.k * self.total_latent_dim))
        # print(to_conv1d.size())  # torch.Size([128, 1, 5238])
        # time.sleep(99)
        conv1d_res = self.conv1d_params1(to_conv1d)
        conv1d_res = self.conv1d_activation(conv1d_res)
        conv1d_res = self.maxpool1d(conv1d_res)
        conv1d_res = self.conv1d_params2(conv1d_res)
        conv1d_res = self.conv1d_activation(conv1d_res)
        to_dense = conv1d_res.view(len(graph_sizes), -1)

        if self.output_dim > 0:
            out_linear = self.out_params(to_dense)
            reluact_fp = self.conv1d_activation(out_linear)
        else:
            reluact_fp = to_dense

        return self.conv1d_activation(reluact_fp)

    def attpooling_embedding(self, node_feat, edge_feat, edge_index, num_edges_set, n2n_sp, e2n_sp, subg_sp, graph_sizes, node_degs):
        """if exists edge feature, concatenate to node feature vector"""
        if edge_feat is not None:
            print("edge_feat is not none.")
            # input_edge_linear = self.w_e2l(edge_feat)
            input_edge_linear = edge_feat
            e2npool_input = gnn_spmm(e2n_sp, input_edge_linear)
            node_feat = torch.cat([node_feat, e2npool_input], 1)

        """graph convolution layers"""
        lv = 0
        cur_message_layer = node_feat
        cat_message_layers = []
        while lv < len(self.latent_dim):
            # gnn_spmm就是矩阵乘法
            n2npool = gnn_spmm(n2n_sp, cur_message_layer) + cur_message_layer  # Y = (A + I) * X = AX+X
            node_linear = self.conv_params[lv](n2npool)  # Y = Y * W
            normalized_linear = node_linear.div(node_degs)  # Y = D^-1 * Y
            cur_message_layer = torch.tanh(normalized_linear)
            cat_message_layers.append(cur_message_layer)
            lv += 1

        cur_message_layer = torch.cat(cat_message_layers, 1)  # 将layers中的张量按第1维度进行拼接,如两个size为[2,3]的张量拼成一个[2,6]
        # print(cur_message_layer.size())  # torch.Size([xxxx, 97]) xxxx与前面的node_feat一致

        """attpooling layer"""
        batch_sortpooling_graphs = torch.zeros(len(graph_sizes), self.k, self.total_latent_dim)  # graph_sizes是每个子图大小

        if torch.cuda.is_available() and isinstance(node_feat.data, torch.cuda.FloatTensor):
            batch_sortpooling_graphs = batch_sortpooling_graphs.cuda()
        batch_sortpooling_graphs = Variable(batch_sortpooling_graphs)  # 将Tensor转换为Variable之后，可以装载梯度信息

        node_accum_count = 0
        edge_accum_count = 0
        for i in range(subg_sp.size()[0]):
            # print("iter:", i)
            # print("graph size is:", graph_sizes[i])
            to_sort = cur_message_layer[node_accum_count: node_accum_count + graph_sizes[i]]
            edge_ind = torch.arange(edge_accum_count, edge_accum_count + num_edges_set[i]).cuda()
            if self.k < graph_sizes[i]:  # 如果子图大小>k,把排在后面的属性舍去
                k = self.k
                per_edge_index = torch.index_select(edge_index, 1, edge_ind)
                # print(per_edge_index)
                # print(per_edge_index.size())
                # time.sleep(100)
                score = self.poolConv(to_sort, per_edge_index).squeeze()
                # print(score)
                batch = torch.zeros([1, graph_sizes[i]]).squeeze()
                batch = batch.type(dtype='torch.LongTensor').cuda()
                topk_indices = topk(score, k/graph_sizes[i], batch)  # topk中具体保留多少个,基于ratio会有一个向上取整操作
                if len(topk_indices) > k:
                    topk_indices += node_accum_count
                    sortpooling_graph = cur_message_layer.index_select(0, topk_indices[:-1])
                else:
                    topk_indices += node_accum_count
                    sortpooling_graph = cur_message_layer.index_select(0, topk_indices)
                # print("topk ind:", topk_indices)
                # time.sleep(100)
                batch_sortpooling_graphs[i] = sortpooling_graph  # 存放每一个池化后的子图
                node_accum_count += graph_sizes[i]
            elif self.k == graph_sizes[i]:
                node_indices = torch.arange(node_accum_count, node_accum_count + graph_sizes[i]).cuda()
                sortpooling_graph = cur_message_layer.index_select(0, node_indices)
                batch_sortpooling_graphs[i] = sortpooling_graph  # 存放每一个池化后的子图
                node_accum_count += graph_sizes[i]
            else:  # 如果图的大小<k,把缺的部分用0补齐
                k = graph_sizes[i]
                node_indices = torch.arange(node_accum_count, node_accum_count + graph_sizes[i]).cuda()
                sortpooling_graph = cur_message_layer.index_select(0, node_indices)
                to_pad = torch.zeros(self.k - k, self.total_latent_dim)
                if torch.cuda.is_available() and isinstance(node_feat.data, torch.cuda.FloatTensor):
                    to_pad = to_pad.cuda()
                to_pad = Variable(to_pad)
                sortpooling_graph = torch.cat((sortpooling_graph, to_pad), 0)
                batch_sortpooling_graphs[i] = sortpooling_graph  # 存放每一个池化后的子图
                node_accum_count += graph_sizes[i]
            edge_accum_count += num_edges_set[i]

        """traditional 1d convlution and dense layers"""
        to_conv1d = batch_sortpooling_graphs.view((-1, 1, self.k * self.total_latent_dim))
        conv1d_res = self.conv1d_params1(to_conv1d)
        conv1d_res = self.conv1d_activation(conv1d_res)
        conv1d_res = self.maxpool1d(conv1d_res)
        conv1d_res = self.conv1d_params2(conv1d_res)
        conv1d_res = self.conv1d_activation(conv1d_res)
        to_dense = conv1d_res.view(len(graph_sizes), -1)

        if self.output_dim > 0:
            out_linear = self.out_params(to_dense)
            reluact_fp = self.conv1d_activation(out_linear)
        else:
            reluact_fp = to_dense

        return self.conv1d_activation(reluact_fp)
