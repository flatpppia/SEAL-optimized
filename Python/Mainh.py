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
import signal

signal.signal(signal.SIGPIPE, signal.SIG_IGN)  # 忽略SIGPIPE信号

parser = argparse.ArgumentParser(description='Link Prediction with SEAL')
parser.add_argument('--d-name', default='pubmed', help='network name')
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
parser.add_argument('--hop', default='auto', metavar='S', help='enclosing subgraph hop number, options: 1, 2,..."auto"')
parser.add_argument('--max-nodes-per-hop', type=int, default=None, help='if > 0, upper bound the nodes per hop by subsampling')
parser.add_argument('--use-embedding', action='store_true', default=False, help='whether to use node2vec')
parser.add_argument('--use-attribute', action='store_true', default=True, help='whether to use node attributes')
parser.add_argument('--save-model', default=False, help='save the final model')
parser.add_argument('--hh', type=float, default=0.1, help='the homophily percent of syn-cora')
parser.add_argument('--sortpooling-k', type=float, default=0.6, help='the k use in sortpooling')
parser.add_argument('--lr', type=float, default=0.0001, help='the learning rate')
parser.add_argument('--n-paral', type=int, default=8, help='the number of parallel')
parser.add_argument('--epoch', type=int, default=128, help='epochs')
parser.add_argument('--rorc', type=int, default=0, help='0=concatenate; 1=replace')
parser.add_argument('--xxx', type=int, default=0, help='set start method')
parser.add_argument('--pk', type=str, default='sort', help='pooling: sort or att')
parser.add_argument('--use-pagerank', default=False)
parser.add_argument('--m-size', type=int, default=3, help='set subgraph min size.')


def get_current_memory_gb():
    pid = os.getpid()
    p = psutil.Process(pid)
    # print(p)进程号
    info = p.memory_full_info()
    # return info.uss / 1024 / 1024 / 1024
    print('当前进程的内存使用：%.4f GB' % (info.uss / 1024 / 1024 / 1024))


def label_statistics(real_labels):
    # real_labels中的1,0指目标连边的同质异质性
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


args = parser.parse_args()
if args.xxx == 1:
    mp.set_start_method('forkserver', force=True)
if __name__ == '__main__':
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    # print(args.cuda)
    # torch.manual_seed(args.seed)  # 设置CPU生成随机数的种子，并返回一个torch.Generator对象
    # if args.cuda:
    #     torch.cuda.manual_seed(args.seed)  # 设置GPU生成随机数的种子
    print(args)

    # random.seed(cmd_args.seed) np.random.seed(cmd_args.seed) torch.manual_seed(cmd_args.seed)
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
        if args.d_name == 'syn-cora':
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
                # 从大到小排序，sorted结果中第一个元组记录着值为1的元素的下标与该元素
                L = [sorted(enumerate(label_list[i]), key=lambda x: x[1], reverse=True)[0][0] for i in range(len(label_list))]
                L = np.array(L)  # 转成所需的形式
                return L
            d_name = 'ind.' + args.d_name
            dataset = PlanetoidData(dataset_str=d_name, dataset_path='data/inddata')
            adj, features, _, _, _, _, _, _, _, _, labels = dataset.load_data(dataset_str=dataset.dataset_str)
            # print(type(adj)) # 'scipy.sparse.csr.csr_matrix'
            # print(type(features)) # 'scipy.sparse.lil.lil_matrix'
            attributes = features.toarray()  # 'numpy.ndarrary'
            single_labels = setlabels(labels)  # 记录每个节点label的一维列表，例如cora的label范围是0~6
            net = adj.tocsc(adj)
            # print(graph_link_homo(net, single_labels))
            # time.sleep(100)
        elif args.d_name=='actor' or args.d_name=='cornell' or args.d_name=='texas' or args.d_name=='wisconsin':
            d_name = args.d_name + '.npz'
            G = load_npzdata(d_name)
            adj = G['adj']
            attributes = G['features']
            print("attributes: ", attributes.shape)
            single_labels = G['labels']
            print("nodes: ", len(single_labels))
            adj = adj.A
            for i in range(len(single_labels)):  # 去自环
                adj[i][i] = 0
            adj = ssp.csr_matrix(adj)
            net = adj.tocsc(adj)
            # net_triu = ssp.tril(net, k=1)
            # print("links: ", net_triu.nnz)
            # AA = net.A
            # print(np.allclose(AA, np.transpose(AA)))
            # time.sleep(100)
        elif args.d_name=='ENGB' or args.d_name=='ES' or args.d_name=='PTBR' or args.d_name=='RU' or args.d_name=='TW':
            twitch_dataset = twitch.load_twitch_dataset(args.d_name)
            adj = twitch_dataset.graph['adj']
            attributes = twitch_dataset.graph['node_feat'].numpy()
            single_labels = twitch_dataset.label.numpy()
            net = adj.tocsc(adj)
        elif args.d_name=='Reed98':
            facebook_dataset = twitch.load_fb100_dataset(args.d_name)
            net = facebook_dataset.graph['adj']
            attributes = facebook_dataset.graph['node_feat'].numpy()
            single_labels = facebook_dataset.label.numpy()
        else:
            args.data_dir = os.path.join(args.file_dir, 'data/{}.mat'.format(args.d_name))
            data = sio.loadmat(args.data_dir)
            net = data['net']
            # print(type(data))
            # print(type(data['net']))
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
    else:
        # use provided train/test positive links, sample negative from net
        train_pos, train_neg, test_pos, test_neg = sample_neg(net, train_pos=train_pos, test_pos=test_pos,
                                            max_train_num=args.max_train, max_test_num=args.max_test,
                                            all_unknown_as_negative=args.all_unknown_as_negative)

    '''Train and apply classifier'''
    A = net.copy()  # the observed network
    A[test_pos[0], test_pos[1]] = 0  # mask test links
    A[test_pos[1], test_pos[0]] = 0  # mask test links
    A.eliminate_zeros()  # make sure the links are masked when using the sparse matrix in scipy-1.3.x
    # print(type(A))  # <class 'scipy.sparse._csc.csc_matrix'>

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

    if args.only_predict:  # no need to use negatives
        # test_pos is a name only, we don't actually know their labels
        _, test_graphs, max_n_label = links2subgraphs(A, None, None, test_pos, None, single_labels, args.n_paral,
                                                      args.hop, args.m_size, args.max_nodes_per_hop, node_information,
                                                      args.no_parallel, args.use_pagerank)
        print('# test: %d' % (len(test_graphs)))
    else:
        train_graphs, test_graphs, max_n_label = links2subgraphs(A, train_pos, train_neg, test_pos, test_neg,
                                                                 single_labels, args.n_paral, args.hop, args.m_size,
                                                                 args.max_nodes_per_hop, node_information,
                                                                 args.no_parallel, args.use_pagerank)
        print('# train G: %d, # test G: %d' % (len(train_graphs), len(test_graphs)))
        # print(max_n_label)
        # time.sleep(100)
        # 查看已使用内存
        # mem = psutil.virtual_memory()
        # print('当前内存已使用：', float(mem.used) / 1024 / 1024 / 1024)
    real_labels = [test_g.link_label for test_g in test_graphs]

    if args.rorc == 0:
        multi_concat(train_graphs, 8)
        multi_concat(test_graphs, 8)
    if args.rorc == 1:
        multi_replace(train_graphs, 8)
        multi_replace(test_graphs, 8)
    if args.rorc == 2:
        pass

    statistics_result = label_statistics(real_labels)
    with open("results_txt/" + args.d_name + "_sort_result_k" + str(int(args.sortpooling_k*10)) + ".txt", "a+") as ff:
        ff.write("Train links:"+str(len(train_pos[0]))+" test links:"+str(len(test_pos[0]))+'\n')
        ff.write("min subgraph size:"+str(args.m_size)+'\n')
        ff.write("train G:" + str(len(train_graphs)) + " test G:" + str(len(test_graphs))+'\n')
        ff.write("batch size:"+str(args.batch_size)+" learning rate:"+str(args.lr)+
                 " rorc:"+str(args.rorc)+" pool:"+args.pk+'\n')
        ff.write("Use personalization pr."+'\n')
        ff.write(str(statistics_result) + '\n')

    '''
    def create_dict(test_graphs, real_labels):
        graph_label_dict = {}
        for i in range(len(test_graphs)):
            graph_label_dict[test_graphs[i]] = real_labels[i]
        print("create dict done.")
    
        return graph_label_dict
    
    gl_dict = create_dict(test_graphs, real_labels)
    # print(gl_dict[test_graphs[1]])
    '''
    # DGCNN configurations
    if args.only_predict:
        with open('data/{}_hyper.pkl'.format(args.d_name), 'rb') as hyperparameters_name:
            saved_cmd_args = pickle.load(hyperparameters_name)
        for key, value in vars(saved_cmd_args).items():  # replace with saved cmd_args
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
        cmd_args.attr_dim = train_graphs[0].node_features.shape[1]  # Classifier初始化的时候里面有attr_dim参数,所以是有影响的
        print("The dim of attributes is:", cmd_args.attr_dim)

    if cmd_args.sortpooling_k <= 1:
        num_nodes_list = sorted([g.num_nodes for g in train_graphs + test_graphs])  # 所有子图含有的节点数量的排序列表
        k_ = int(math.ceil(cmd_args.sortpooling_k * len(num_nodes_list))) - 1  # 通过s_k*所有子图的总数=下标k_
        cmd_args.sortpooling_k = max(1, num_nodes_list[k_])  # 真正的sortpooling_k要么是10，要么是num_nodes_list中第k_个值
        # print('k used in SortPooling is: ' + str(cmd_args.sortpooling_k))
        with open("results_txt/" + args.d_name + "_sort_result_k" + str(int(args.sortpooling_k*10)) + ".txt", "a+") as ff:
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
        random.shuffle(train_idxes)  # 下标被打乱
        classifier.train()
        avg_loss, _ = loop_dataset(train_graphs, classifier, train_idxes, optimizer=optimizer, bsize=args.batch_size)
        if not cmd_args.printAUC:
            avg_loss[2] = 0.0
        print('\033[92maverage training of epoch %d: loss %.5f acc %.5f auc %.5f ap %.5f\033[0m'
              % (epoch, avg_loss[0], avg_loss[1], avg_loss[2], avg_loss[3]))
        # # 查看已使用内存
        # mem = psutil.virtual_memory()
        # print('当前内存已使用：', float(mem.used) / 1024 / 1024 / 1024)

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
            # time.sleep(3)

            count_result = compare_result(real_labels, total_pre_result)
            if not cmd_args.printAUC:
                test_loss[2] = 0.0
            print('\033[94maverage test of epoch %d: loss %.5f acc %.5f auc %.5f ap %.5f\033[0m'
                  % (epoch, test_loss[0], test_loss[1], test_loss[2], test_loss[3]))
            # print(statistics_result)
            print(count_result)  # 把比较结果存进txt会更方便
            with open("results_txt/" + args.d_name + "_sort_result_k" + str(int(args.sortpooling_k*10)) + ".txt", "a+") as ff:
                ff.write(str(epoch) + ': ')
                ff.write(str(count_result))
                ff.write('test of epoch %d: loss %.5f acc %.5f auc %.5f ap %.5f'
                         % (epoch, test_loss[0], test_loss[1], test_loss[2], test_loss[3]))
                ff.write('\n')

    with open("results_txt/" + args.d_name + "_sort_result_k" + str(int(args.sortpooling_k*10)) + ".txt", "a+") as ff:
        ff.write('Final test performance: epoch %d: loss %.5f acc %.5f auc %.5f ap %.5f'
                 % (best_epoch, test_loss[0], test_loss[1], test_loss[2], test_loss[3]))
        ff.write('\n')
    print('\033[95mFinal test performance: epoch %d: loss %.5f acc %.5f auc %.5f ap %.5f\033[0m'
          % (best_epoch, test_loss[0], test_loss[1], test_loss[2], test_loss[3]))

    if args.save_model:
        model_name = 'model/{}_model.pth'.format(args.d_name)
        print('Saving final model states to {}...'.format(model_name))
        torch.save(classifier.state_dict(), model_name)
        hyper_name = 'data/{}_hyper.pkl'.format(args.d_name)
        with open(hyper_name, 'wb') as hyperparameters_file:
            pickle.dump(cmd_args, hyperparameters_file)
            print('Saving hyperparameters to {}...'.format(hyper_name))
