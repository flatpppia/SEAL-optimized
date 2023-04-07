from torch_geometric.nn import GCNConv
from torch_geometric.nn.pool.topk_pool import topk, filter_adj
from torch.nn import Parameter
import torch
import time


class SAGPool(torch.nn.Module):
    def __init__(self, in_channels, ratio=0.8, Conv=GCNConv, non_linearity=torch.tanh):  # tanh将元素调整到区间(-1,1)内

        super(SAGPool, self).__init__()  # super().__init__()的作用是执行父类的构造函数，使得我们能够调用父类的私有属性
        self.in_channels = in_channels
        self.ratio = ratio
        self.score_layer = Conv(in_channels, 1)  # 输入128输出1的卷积层
        # in_channels是节点特征的维度, out_channels是设定的降维维度.输入(N, in_channels),输出(N, out_channels),N是节点数
        self.non_linearity = non_linearity

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # print(x)  # torch.Size([2732, 128])
        # time.sleep(100)
        # x has shape [N, in_channels], class'torch.Tensor'
        # edge_index has shape [2, E]
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))  # 通过new构造一个数据大小、类型相同的新张量
        # x = x.unsqueeze(-1) if x.dim() == 1 else x
        # print("score:", self.score_layer(x, edge_index).size())  # torch.Size([2732, 1])
        score = self.score_layer(x, edge_index).squeeze()  # 移除数组中维度为1的维度,torch.Size([2732])
        print("score: ", score.size())  # Size([2732])
        # print(self.ratio)
        perm = topk(score, 20, batch)
        print(perm)
        print("perm: ", perm.size())  # Size([1408])
        # perm中的是池化后的节点下标
        time.sleep(1000)
        # yy = x[perm]  # torch.Size([1408, 128])
        # nl = self.non_linearity(score[perm])  # torch.Size([1408])
        # xy = self.non_linearity(score[perm]).view(-1, 1)  # torch.Size([1408, 1])
        x = x[perm] * self.non_linearity(score[perm]).view(-1, 1)  # view的作用相当于numpy中的reshape,重新定义矩阵的形状
        # print(x.size())
        # time.sleep(1000)
        batch = batch[perm]
        edge_index, edge_attr = filter_adj(edge_index, edge_attr, perm, num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm
