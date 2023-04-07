import torch
# from torch_geometric.datasets import TUDataset
from tudataset import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric import utils
from networks import Net
import torch.nn.functional as F
import argparse
import os
import time
from torch.utils.data import random_split
parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=777, help='seed')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
parser.add_argument('--nhid', type=int, default=128, help='hidden size')
parser.add_argument('--pooling_ratio', type=float, default=0.6, help='pooling ratio')
parser.add_argument('--dropout_ratio', type=float, default=0.5, help='dropout ratio')
parser.add_argument('--dataset', type=str, default='MUTAG', help='DD/PROTEINS/NCI1/NCI109/Mutagenicity')
parser.add_argument('--epochs', type=int, default=100000, help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=50, help='patience for earlystopping')
parser.add_argument('--pooling_layer_type', type=str, default='GCNConv', help='DD/PROTEINS/NCI1/NCI109/Mutagenicity')
# MUTAG:188
args = parser.parse_args()
args.device = 'cpu'
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    args.device = 'cuda:0'

datap = os.path.join('data', args.dataset)
print(datap)
dataset = TUDataset(datap, name=args.dataset)
args.num_classes = dataset.num_classes
args.num_features = dataset.num_features

# print(len(dataset))
num_training = int(len(dataset)*0.8)
num_val = int(len(dataset)*0.1)
num_test = len(dataset) - (num_training+num_val)
training_set, validation_set, test_set = random_split(dataset, [num_training, num_val, num_test])
# print(type(training_set))  # <class 'torch.utils.data.dataset.Subset'>
# print(training_set[0])  # Data(edge_index=[2, 50], x=[22, 7], edge_attr=[50, 4], y=[1])
# print(training_set[1])  # Data(edge_index=[2, 54], x=[23, 7], edge_attr=[54, 4], y=[1])
# print(training_set[77])  # Data(edge_index=[2, 20], x=[10, 7], edge_attr=[20, 4], y=[1])
# print(training_set[78])  # Data(edge_index=[2, 34], x=[15, 7], edge_attr=[34, 4], y=[1])
# print(training_set[79])  # Data(edge_index=[2, 26], x=[13, 7], edge_attr=[26, 4], y=[1])
# print("training_set", len(training_set))  # 150
# print("validation_set", len(validation_set))  # 18
# print("test_set", len(test_set))  # 20
'''
look_set = training_set[77:80]
print(look_set[0].edge_index)
print(look_set[1].edge_index)
look_loder = DataLoader(look_set, batch_size=3, shuffle=False)
for i, data in enumerate(look_loder):
    print(data.edge_index)
time.sleep(100)
'''
train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
# print(type(train_loader))  # class 'torch_geometric.loader.dataloader.DataLoader'
# print(train_loader.__dict__)
# print(type(train_loader.dataset))  # class 'torch.utils.data.dataset.Subset'
# print(train_loader.dataset.__dict__)
# time.sleep(100)
# ddata = list(enumerate(train_loader))
# print(ddata[0][1])
# print(ddata[0][1].batch.size())  # tensor([  0,   0,   0,  ..., 149, 149, 149])
# print(ddata[0][1].batch[15])
# print(ddata[0][1].batch[21])
# print(ddata[0][1].batch[22])
# time.sleep(90)
model = Net(args).to(args.device)  # 将模型放到GPU上跑
# print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def test(model, loader):
    model.eval()
    correct = 0.
    loss = 0.
    for data in loader:
        data = data.to(args.device)
        out = model(data)
        pred = out.max(dim=1)[1]
        # 使用torch.max()函数时，生成的张量会比原来的维度减少一维(除非原来的张量只有一维).要减少的是哪一维由dim参数决定，dim参数实际上指的是我们计算过程中所要消去的维度。
        correct += pred.eq(data.y).sum().item()
        loss += F.nll_loss(out, data.y, reduction='sum').item()
    return correct / len(loader.dataset), loss / len(loader.dataset)


min_loss = 1e10
patience = 0

for epoch in range(args.epochs):
    model.train()
    for i, data in enumerate(train_loader):  # enumerate可以同时获得索引和值
        data = data.to(args.device)
        # print(type(data))  # 'torch_geometric.data.batch.DataBatch'
        # DataBatch(edge_index=[2, 6042], x=[2732, 7], edge_attr=[6042, 4], y=[150], batch=[2732], ptr=[151])
        # print(data.__dict__)
        # time.sleep(90)
        out = model(data)
        loss = F.nll_loss(out, data.y)
        print("Training loss:{}".format(loss.item()))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    val_acc, val_loss = test(model, val_loader)
    print("Validation loss:{}\taccuracy:{}".format(val_loss, val_acc))
    if val_loss < min_loss:
        torch.save(model.state_dict(), 'latest.pth')
        print("Model saved at epoch{}".format(epoch))
        min_loss = val_loss
        patience = 0
    else:
        patience += 1
    if patience > args.patience:
        print("Reach the patience for earlystopping")
        break 

model = Net(args).to(args.device)
model.load_state_dict(torch.load('latest.pth'))
test_acc, test_loss = test(model, test_loader)
print(f'Test accuarcy:{test_acc}')
# print("Test accuarcy:{}".fotmat(test_acc))
