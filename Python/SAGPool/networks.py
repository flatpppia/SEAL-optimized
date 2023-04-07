import torch
from torch_geometric.nn import GCNConv
# from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from layers import SAGPool
import time

# 把定义的网络模型当作函数调用的时候就会自动调用定义的网络模型的forward方法
# 通过nn.Module的__call__方法调用实现
# 相当于调用了模型就是直接调用它的forward函数,y=model(x),这个x就是直接传入到forward函数的x参数


class Net(torch.nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid  # hidden size = 128
        self.num_classes = args.num_classes
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio
        
        self.conv1 = GCNConv(self.num_features, self.nhid)  # size[num_features, 128]
        self.pool1 = SAGPool(self.nhid, ratio=self.pooling_ratio)
        self.conv2 = GCNConv(self.nhid, self.nhid)
        self.pool2 = SAGPool(self.nhid, ratio=self.pooling_ratio)
        self.conv3 = GCNConv(self.nhid, self.nhid)
        self.pool3 = SAGPool(self.nhid, ratio=self.pooling_ratio)

        self.lin1 = torch.nn.Linear(self.nhid*2, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid//2)
        self.lin3 = torch.nn.Linear(self.nhid//2, self. num_classes)

    def forward(self, data):
        # print(type(data))  # 'torch_geometric.data.batch.DataBatch'
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # print(batch.size())  # 2732
        # print("xsize0: ", x.size())  # [2732,7]
        # print(edge_index.size())  # torch.Size([2, 6042])

        x = F.relu(self.conv1(x, edge_index))
        # print("xsize1: ", x.size())  # torch.Size([2732, 128])
        x, edge_index, _, batch, _ = self.pool1(x, edge_index, None, batch)
        # print("xsize2: ", x.size())  # torch.Size([1408, 128])
        # print(edge_index)  # torch.Size([2, 2710])
        # print(batch)
        # time.sleep(100)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x
