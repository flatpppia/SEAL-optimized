import sys
import os
import torch
import random
import time
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import pdb
from DGCNN_embedding import DGCNN
from mlp_dropout import MLPClassifier, MLPRegression, MLPClass
from sklearn import metrics
from util import cmd_args, load_data


class Classifier(nn.Module):
    def __init__(self, regression=False):
        super(Classifier, self).__init__()  # super().__init__()的作用是执行父类的构造函数，使得我们能够调用父类的私有属性
        self.regression = regression
        if cmd_args.gm == 'DGCNN':
            model = DGCNN
        else:
            print('unknown gm %s' % cmd_args.gm)
            sys.exit()

        if cmd_args.gm == 'DGCNN':
            self.gnn = model(latent_dim=cmd_args.latent_dim,
                            output_dim=cmd_args.out_dim,
                            num_node_feats=cmd_args.feat_dim+cmd_args.attr_dim,
                            num_edge_feats=cmd_args.edge_feat_dim,
                            k=cmd_args.sortpooling_k, 
                            conv1d_activation=cmd_args.conv1d_activation)
            # print(self.gnn)
        out_dim = cmd_args.out_dim
        # print("outdim:", out_dim)  # 0
        if out_dim == 0:
            if cmd_args.gm == 'DGCNN':
                out_dim = self.gnn.dense_dim
            else:
                out_dim = cmd_args.latent_dim
        self.mlp = MLPClassifier(input_size=out_dim, hidden_size=cmd_args.hidden, num_class=cmd_args.num_class, with_dropout=cmd_args.dropout)
        # self.mlp = MLPClass(input_size=out_dim, hidden_size=cmd_args.hidden, num_class=cmd_args.num_class, with_dropout=cmd_args.dropout)
        if regression:
            self.mlp = MLPRegression(input_size=out_dim, hidden_size=cmd_args.hidden, with_dropout=cmd_args.dropout)

    def PrepareFeatureLabel(self, batch_graph):
        if self.regression:
            labels = torch.FloatTensor(len(batch_graph))
        else:
            labels = torch.LongTensor(len(batch_graph))
            lk_labels = torch.LongTensor(len(batch_graph))  # 需要一个记录目标连边的同质异质性的列表
        n_nodes = 0

        if batch_graph[0].node_tags is not None:
            node_tag_flag = True
            concat_tag = []
        else:
            node_tag_flag = False
        if batch_graph[0].node_features is not None:
            node_feat_flag = True
            concat_feat = []
        else:
            node_feat_flag = False
        if cmd_args.edge_feat_dim > 0:
            edge_feat_flag = True
            concat_edge_feat = []
        else:
            edge_feat_flag = False

        # batch_tmp = torch.Tensor([])  # 首先创建一个空tensor
        for i in range(len(batch_graph)):
            labels[i] = batch_graph[i].label  # 这里的label是g_label
            lk_labels[i] = batch_graph[i].link_label  # 这里的label是目标连边的同质异质性
            n_nodes += batch_graph[i].num_nodes
            if node_tag_flag == True:
                concat_tag += batch_graph[i].node_tags
            if node_feat_flag == True:
                tmp = torch.from_numpy(batch_graph[i].node_features).type('torch.FloatTensor')
                # batch_tmp = torch.cat([batch_tmp, torch.zeros([1, batch_graph[i].num_nodes])+i], 1)  # 按照topk的需求制作batch信息,这样不行,不要了
                concat_feat.append(tmp)
            if edge_feat_flag == True:
                if batch_graph[i].edge_features is not None:  # in case no edge in graph[i]
                    tmp = torch.from_numpy(batch_graph[i].edge_features).type('torch.FloatTensor')
                    concat_edge_feat.append(tmp)
        # batch_tmp = batch_tmp.squeeze().type('torch.LongTensor')  # 维度改成torch.Size([x]),且需要指定数据格式

        if node_tag_flag == True:
            concat_tag = torch.LongTensor(concat_tag).view(-1, 1)
            node_tag = torch.zeros(n_nodes, cmd_args.feat_dim)
            node_tag.scatter_(1, concat_tag, 1)

        if node_feat_flag == True:
            node_feat = torch.cat(concat_feat, 0)

        if node_feat_flag and node_tag_flag:
            # concatenate one-hot embedding of node tags (node labels) with continuous node features
            node_feat = torch.cat([node_tag.type_as(node_feat), node_feat], 1)
        elif node_feat_flag == False and node_tag_flag == True:
            node_feat = node_tag
        elif node_feat_flag == True and node_tag_flag == False:
            pass
        else:
            node_feat = torch.ones(n_nodes, 1)  # use all-one vector as node features
        
        if edge_feat_flag == True:
            edge_feat = torch.cat(concat_edge_feat, 0)

        if cmd_args.mode == 'gpu':
            node_feat = node_feat.cuda()
            labels = labels.cuda()
            lk_labels = lk_labels.cuda()
            # batch_tmp = batch_tmp.cuda()
            if edge_feat_flag == True:
                edge_feat = edge_feat.cuda()

        if edge_feat_flag == True:
            return node_feat, edge_feat, labels#, batch_tmp
        return node_feat, labels, lk_labels#, batch_tmp
        # return node_feat, labels

    def forward(self, batch_graph):
        feature_label = self.PrepareFeatureLabel(batch_graph)
        '''
        if len(feature_label) == 2:
            node_feat, labels = feature_label
            edge_feat = None
        elif len(feature_label) == 3:
            node_feat, edge_feat, labels = feature_label
        embed = self.gnn(batch_graph, node_feat, edge_feat)
        '''
        if len(feature_label) == 3:
            node_feat, labels, link_labels = feature_label
            edge_feat = None
        elif len(feature_label) == 4:
            node_feat, edge_feat, labels, link_labels = feature_label
        embed = self.gnn(batch_graph, node_feat, edge_feat)

        return self.mlp(embed, link_labels, labels)  # link_labels对应forward里的z
        # return self.mlp(embed, labels)

    def output_features(self, batch_graph):
        feature_label = self.PrepareFeatureLabel(batch_graph)

        if len(feature_label) == 2:
            node_feat, labels = feature_label
            edge_feat = None
        elif len(feature_label) == 3:
            node_feat, edge_feat, labels = feature_label
        embed = self.gnn(batch_graph, node_feat, edge_feat)
        '''不要了
        if len(feature_label) == 3:
            node_feat, labels, batch_index = feature_label
            edge_feat = None
        elif len(feature_label) == 4:
            node_feat, edge_feat, labels, batch_index = feature_label
        embed = self.gnn(batch_graph, node_feat, edge_feat, batch_index)
        '''
        return embed, labels


def compare_result(real_label, pred_result):
    tongzhi_pos_count = 0  # 同质性正连边被预测出来的统计
    yizhi_pos_count = 0  # 异质性正连边被预测出来的统计
    tongzhi_neg_count = 0
    yizhi_neg_count = 0
    iter = int(len(pred_result)/2)
    for i in range(iter):
        if (pred_result[i][0]==True) and (real_label[i] == 1):
            tongzhi_pos_count += 1
        elif (pred_result[i][0]==True) and (real_label[i] == 0):
            yizhi_pos_count += 1
    for j in range(iter, iter*2):
        if (pred_result[j][0]==True) and (real_label[j] == 1):
            tongzhi_neg_count += 1
        elif (pred_result[j][0]==True) and (real_label[j] == 0):
            yizhi_neg_count += 1
    result = {"同质连边1": tongzhi_pos_count, "异质连边1": yizhi_pos_count, "同质连边0": tongzhi_neg_count, "异质连边0": yizhi_neg_count}

    return result


# import pdb
def loop_dataset(g_list, classifier, sample_idxes, optimizer=None, bsize=cmd_args.batch_size):
    # print("batch size is:", bsize)
    total_loss = []
    total_iters = (len(sample_idxes) + (bsize - 1) * (optimizer is None)) // bsize
    pbar = tqdm(range(total_iters), unit='batch')
    all_targets = []
    all_scores = []

    n_samples = 0
    total_p_result = []
    for pos in pbar:
        selected_idx = sample_idxes[pos * bsize: (pos + 1) * bsize]  # 每次选择一个batchsize的量

        batch_graph = [g_list[idx] for idx in selected_idx]
        targets = [g_list[idx].label for idx in selected_idx]
        all_targets += targets
        if classifier.regression:
            # print("regression is T.")
            pred, mae, loss = classifier(batch_graph)
            all_scores.append(pred.cpu().detach())  # for binary classification
        else:
            # pdb.set_trace()
            logits, loss, acc, pre_result = classifier(batch_graph)
            pre_result = pre_result.cpu().numpy().tolist()
            # print(pre_result)
            total_p_result += pre_result
            # print("total pre results:", len(total_p_result))
            all_scores.append(logits[:, 1].cpu().detach())  # for binary classification

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = loss.data.cpu().detach().numpy()
        if classifier.regression:
            # pbar.update(1)
            pbar.set_description('MSE_loss: %0.5f MAE_loss: %0.5f' % (loss, mae))
            total_loss.append(np.array([loss, mae]) * len(selected_idx))
        else:
            pbar.update(1)
            pbar.set_description('Loss: %0.5f Acc: %0.5f' % (loss, acc))
            total_loss.append(np.array([loss, acc]) * len(selected_idx))

        n_samples += len(selected_idx)

    if optimizer is None:
        assert n_samples == len(sample_idxes)
    total_loss = np.array(total_loss)
    avg_loss = np.sum(total_loss, 0) / n_samples
    all_scores = torch.cat(all_scores).cpu().numpy()
    # np.savetxt('test_scores.txt', all_scores)  # output test predictions
    
    if not classifier.regression and cmd_args.printAUC:
        all_targets = np.array(all_targets)
        fpr, tpr, _ = metrics.roc_curve(all_targets, all_scores, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        ap = metrics.average_precision_score(all_targets, all_scores)
        avg_loss = np.concatenate((avg_loss, [auc], [ap]))
    else:
        avg_loss = np.concatenate((avg_loss, [0.0]))
    
    return avg_loss, total_p_result

'''
if __name__ == '__main__':
    print(cmd_args)
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)

    train_graphs, test_graphs = load_data()
    print('# train: %d, # test: %d' % (len(train_graphs), len(test_graphs)))

    if cmd_args.sortpooling_k <= 1:
        num_nodes_list = sorted([g.num_nodes for g in train_graphs + test_graphs])
        cmd_args.sortpooling_k = num_nodes_list[int(math.ceil(cmd_args.sortpooling_k * len(num_nodes_list))) - 1]
        cmd_args.sortpooling_k = max(10, cmd_args.sortpooling_k)
        print('k used in SortPooling is: ' + str(cmd_args.sortpooling_k))

    classifier = Classifier()
    if cmd_args.mode == 'gpu':
        classifier = classifier.cuda()

    optimizer = optim.Adam(classifier.parameters(), lr=cmd_args.learning_rate)

    train_idxes = list(range(len(train_graphs)))
    best_loss = None
    for epoch in range(cmd_args.num_epochs):
        random.shuffle(train_idxes)
        classifier.train()
        avg_loss = loop_dataset(train_graphs, classifier, train_idxes, optimizer=optimizer)
        if not cmd_args.printAUC:
            avg_loss[2] = 0.0
        print('\033[92maverage training of epoch %d: loss %.5f acc %.5f auc %.5f\033[0m' % (epoch, avg_loss[0], avg_loss[1], avg_loss[2]))

        classifier.eval()
        test_loss = loop_dataset(test_graphs, classifier, list(range(len(test_graphs))))
        if not cmd_args.printAUC:
            test_loss[2] = 0.0
        print('\033[93maverage test of epoch %d: loss %.5f acc %.5f auc %.5f\033[0m' % (epoch, test_loss[0], test_loss[1], test_loss[2]))

    with open(cmd_args.data + '_acc_results.txt', 'a+') as f:
        f.write(str(test_loss[1]) + '\n')

    if cmd_args.printAUC:
        with open(cmd_args.data + '_auc_results.txt', 'a+') as f:
            f.write(str(test_loss[2]) + '\n')

    if cmd_args.extract_features:
        features, labels = classifier.output_features(train_graphs)
        labels = labels.type('torch.FloatTensor')
        np.savetxt('extracted_features_train.txt', torch.cat([labels.unsqueeze(1), features.cpu()], dim=1).detach().numpy(), '%.4f')
        features, labels = classifier.output_features(test_graphs)
        labels = labels.type('torch.FloatTensor')
        np.savetxt('extracted_features_test.txt', torch.cat([labels.unsqueeze(1), features.cpu()], dim=1).detach().numpy(), '%.4f')
'''
