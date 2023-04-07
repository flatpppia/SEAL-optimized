from __future__ import print_function
import os
import sys
import numpy as np
import torch
import random
import time
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import pdb

sys.path.append('%s/lib' % os.path.dirname(os.path.realpath(__file__)))
from pytorch_util import weights_init


class MLPRegression(nn.Module):
    def __init__(self, input_size, hidden_size, with_dropout=False):
        super(MLPRegression, self).__init__()

        self.h1_weights = nn.Linear(input_size, hidden_size)
        self.h2_weights = nn.Linear(hidden_size, 1)
        self.with_dropout = with_dropout

        weights_init(self)

    def forward(self, x, y= None):
        h1 = self.h1_weights(x)
        h1 = F.relu(h1)

        if self.with_dropout:
            h1 = F.dropout(h1, training=self.training)
        pred = self.h2_weights(h1)[:, 0]

        if y is not None:
            y = Variable(y)
            mse = F.mse_loss(pred, y)
            mae = F.l1_loss(pred, y)
            mae = mae.cpu().detach()
            return pred, mae, mse
        else:
            return pred


class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_class, with_dropout=False):
        super(MLPClassifier, self).__init__()

        self.h1_weights = nn.Linear(input_size, hidden_size)
        self.h2_weights = nn.Linear(hidden_size, num_class)
        self.with_dropout = with_dropout

        weights_init(self)

    def forward(self, x, z, y=None):
        # x是embeding, y是g_label的列表, z是目标连边同质异质性列表
        h1 = self.h1_weights(x)
        h1 = F.relu(h1)
        if self.with_dropout:
            # print('have dropout.')
            h1 = F.dropout(h1, training=self.training)

        logits = self.h2_weights(h1)
        logits = F.log_softmax(logits, dim=1)  # 在softmax的结果上做一次log运算,1是行,0是列
        # softmax + NLLLoss = CrossEntropyLoss,交叉熵
        if y is not None:
            y = Variable(y)
            z = Variable(z)
            # print(logits.size()) torch.Size([128, 2]) 由于标签是0或1,所以会有个2选1的操作,再将选出的值去负求和求均值,得到loss
            # print(y.size())  torch.Size([128])
            total_loss = F.nll_loss(logits, y)
            pred = logits.data.max(1, keepdim=True)[1]  # 括号里的1代表查找第二维中的最大值，keepdim=true是对应维度被变成1
            # print(pred.size())  # torch.Size([128, 1])
            yizhi_zheng_ind = [i for i in range(x.size()[0]) if y[i] == 1 and z[i] == 0]
            if len(yizhi_zheng_ind) != 0:
                yz_logits = logits[yizhi_zheng_ind]
                yz_y = y[yizhi_zheng_ind]
                yz_loss = F.nll_loss(yz_logits, yz_y)
                total_loss = 0.7 * total_loss + 0.3 * yz_loss
            # eq()函数作用为返回一个和pred相同大小的tensor, 与目标值相等的位置为T, 其余为F
            pred_result = pred.eq(y.data.view_as(pred))
            # print(pred.eq(y.data.view_as(pred)))
            # print(y.size()[0])
            acc = pred.eq(y.data.view_as(pred)).cpu().sum().item() / float(y.size()[0])  # y.size即batch_size
            return logits, total_loss, acc, pred_result
        else:
            return logits


class MLPClass(nn.Module):
    def __init__(self, input_size, hidden_size, num_class, with_dropout=False):
        super(MLPClass, self).__init__()

        self.h1_weights = nn.Linear(input_size, hidden_size)
        self.h2_weights = nn.Linear(hidden_size, num_class)
        self.with_dropout = with_dropout

        weights_init(self)

    def forward(self, x, y=None):
        # x是embeding, y是g_label的列表
        h1 = self.h1_weights(x)
        h1 = F.relu(h1)
        if self.with_dropout:
            h1 = F.dropout(h1, training=self.training)

        logits = self.h2_weights(h1)
        logits = F.log_softmax(logits, dim=1)  # 在softmax的结果上做一次log运算,1是行,0是列
        # softmax + NLLLoss = CrossEntropyLoss,交叉熵
        if y is not None:
            y = Variable(y)
            loss = F.nll_loss(logits, y)
            pred = logits.data.max(1, keepdim=True)[1]  # 括号里的1代表查找第二维中的最大值，keepdim=true是对应维度被变成1
            # eq()函数作用为返回一个和pred相同大小的tensor, 与目标值相等的位置为T, 其余为F
            pred_result = pred.eq(y.data.view_as(pred))
            acc = pred.eq(y.data.view_as(pred)).cpu().sum().item() / float(y.size()[0])  # y.size即batch_size
            return logits, loss, acc, pred_result
        else:
            return logits

# cnn进行前向传播阶段，依次调用每个Layer的Forward函数，得到逐层的输出，最后一层与目标函数比较得到损失函数，
# 计算误差更新值，通过反向传播逐层到达第一层，所有权值在反向传播结束时一起更新。
