# encoding:utf-8
from ast import arg
import sys
import time
import argparse
import numpy as np
import pandas as pd
import random
import math
import datetime
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.utils.data
import torch.optim as optim
from sklearn.model_selection import train_test_split   # split包
from sklearn.metrics import roc_auc_score, f1_score

from ktMaxDiffuCl.model import ktMaxDiffuCl
from ktMaxDiffuCl.metrics import *
from ktMaxDiffuCl.gaussian_diffusion import *
from ktMaxDiffuCl.data_loader import load_data


import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2'

np.random.seed(222)
 
# prepare arguments (hyperparameters)
parser = argparse.ArgumentParser() 
parser.add_argument('--model_name', type=str, default='ktMaxDiffuCl', help='name of model')
parser.add_argument('--dataset', type=str, default='music_v1', help='which dataset to use')

parser.add_argument('--n_epochs', type=int, default=200, help='the number of epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--ratio', type=float, default=0.8, help='size of training dataset')

parser.add_argument('--aggregator', type=str, default='sum', help='which aggregator to use')
parser.add_argument('--neighbor_sample_size', type=int, default=8, help='the number of neighbors to be sampled')
parser.add_argument('--dim', type=int, default=16, help='dimension of user and entity embeddings')
parser.add_argument('--n_iter', type=int, default=1, help='number of iterations when computing entity representation')
parser.add_argument('--l2_weight', type=float, default=0.0001, help='weight of l2 regularization')

parser.add_argument('--click_sequence_size', type=int, default=128, help='size of user click item sequence')
parser.add_argument('--rnn_input_size', type=int, default=16, help='size of rnn input size, embedding_dim')
parser.add_argument('--rnn_hidden_size', type=int, default=8, help='size of rnn hidden size')
parser.add_argument('--rnn_num_layers', type=int, default=1, help='size of rnn number layers')

parser.add_argument('--test_start_epoch', type=int, default=45, help='test start epoch')
parser.add_argument('--test_step_len', type=int, default=4, help='test start epoch')

parser.add_argument("--gpu_id", type=int, default=1, help="GPU device ID to use")

parser.add_argument('--num_steps', type=int, default=100, help='加噪的步数')
args = parser.parse_args()


# Dataset class  数据的np.array
class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        user_id = np.array(self.df.iloc[idx]['userID'])
        item_id = np.array(self.df.iloc[idx]['itemID'])
        label = np.array(self.df.iloc[idx]['label'], dtype=np.float32)
        return user_id, item_id, label

def main():
    print('arguments...')
    for k,v in sorted(vars(args).items()):
        print(k,'=',v)
    
    data = load_data(args)
    n_user, n_item, n_entity, n_relation = data[0], data[1], data[2], data[3]
    train_data, eval_data, test_data = data[4], data[5], data[6]
    adj_entity, adj_relation = data[7], data[8]
    
    print('print data...')
    print('n_user:', n_user)
    print('n_item:', n_item)
    print('n_entity:', n_entity)
    print('n_relation:', n_relation)
    print('len_train_data:', len(train_data))
    print('len_eval_data:', len(eval_data))
    print('len_test_data:', len(test_data))

    df_train_data = numpy2dataframe(train_data)
    df_eval_data = numpy2dataframe(eval_data)
    df_test_data = numpy2dataframe(test_data)

    train_dataset = Dataset(df_train_data)
    eval_dataset = Dataset(df_eval_data)
    test_dataset = Dataset(df_test_data)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)  # 划分batch
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size)  # 划分batch
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)
   
    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(args.gpu_id))
    else:
        device = torch.device("cpu")
    # device = torch.device('cpu')
    print('device: ', device)

    adj_entity = torch.from_numpy(adj_entity).to(device=device)
    adj_relation = torch.from_numpy(adj_relation).to(device=device)

    # top-K evaluation settings
    user_list, train_record, test_record, item_set, k_list, user_click_item_pos = topk_settings(train_data, test_data, n_item)
    print('user_list', len(user_list))
    print('train_record', len(train_record))
    print('test_record', len(test_record))
    print('item_set', len(item_set))
    print('k_list', len(k_list))
    # sys.exit(0)

    ### Embedding
    usr_emb = torch.nn.Embedding(n_user, args.dim).to(device)
    ent_emb = torch.nn.Embedding(n_entity, args.dim).to(device)
    rel_emb = torch.nn.Embedding(n_relation, args.dim).to(device)

    model = ktMaxDiffuCl(usr_emb, ent_emb, rel_emb, adj_entity, adj_relation, user_click_item_pos, args, device).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_weight)

    print('Start Train...')
    epoch_list, train_loss_list, test_loss_list = [], [], []
    R_at1_list, R_at2_list, R_at5_list, R_at10_list, R_at20_list, R_at50_list, R_at100_list = [], [], [], [], [], [], []
    N_at1_list, N_at2_list, N_at5_list, N_at10_list, N_at20_list, N_at50_list, N_at100_list = [], [], [], [], [], [], []
    P_at1_list, P_at2_list, P_at5_list, P_at10_list, P_at20_list, P_at50_list, P_at100_list = [], [], [], [], [], [], []

    epoch = 0
    for epoch in range(args.n_epochs):
        train_loss = 0.0
        for i, (user_ids, item_ids, labels) in enumerate(train_loader):
            model.train()
            user_ids, item_ids, labels = user_ids.to(device), item_ids.to(device), labels.to(device)
            if len(user_ids) == args.batch_size:  # 确保长度一致

                outloss = model.forward_pretrain(user_ids, item_ids, labels)  # 模型输出预测值
                outloss.backward()
                optimizer.step()
                optimizer.zero_grad()

                train_loss += outloss.item()

        train_loss = train_loss / len(train_loader)

        with torch.no_grad():
            model.eval()
            # train_loss, train_auc, train_f1 = ctr_eval(model, criterion, train_loader, device)
            # eval_loss, eval_auc, eval_f1 = ctr_eval(model, criterion, eval_loader, device)
            test_loss = ctr_eval(model, criterion, test_loader, device)
            
            print('Epoch {} | train_loss {:.4f} test_loss {:.4f}'.format(epoch, train_loss, test_loss))

            if epoch > args.test_start_epoch and epoch % args.test_step_len == 0:

                epoch_list.append(epoch)
                train_loss_list.append(train_loss)
                # train_auc_list.append(train_auc)
                # train_f1_list.append(train_f1)

                # eval_loss_list.append(eval_loss)
                # eval_auc_list.append(eval_auc)
                # eval_f1_list.append(eval_f1)

                test_loss_list.append(test_loss)
                # test_auc_list.append(test_auc)
                # test_f1_list.append(test_f1)

                precision, recall, ndcg = topk_eval(model, user_list, train_record, test_record, item_set, k_list, args.batch_size, device)

                P_at1_list.append(precision[0])
                P_at2_list.append(precision[1])
                P_at5_list.append(precision[2])
                P_at10_list.append(precision[3])
                P_at20_list.append(precision[4])
                P_at50_list.append(precision[5])
                P_at100_list.append(precision[6])

                R_at1_list.append(recall[0])
                R_at2_list.append(recall[1])
                R_at5_list.append(recall[2])
                R_at10_list.append(recall[3])
                R_at20_list.append(recall[4])
                R_at50_list.append(recall[5])
                R_at100_list.append(recall[6])

                N_at1_list.append(ndcg[0])
                N_at2_list.append(ndcg[1])
                N_at5_list.append(ndcg[2])
                N_at10_list.append(ndcg[3])
                N_at20_list.append(ndcg[4])
                N_at50_list.append(ndcg[5])
                N_at100_list.append(ndcg[6])

                print('precision: @1 @2 @5 @10 @20 @50 @100 \n', end='')
                for i in precision:
                    print('%.4f\t' % i, end='')
                print()
                print('recall   : @1 @2 @5 @10 @20 @50 @100 \n', end='')
                for i in recall:
                    print('%.4f\t' % i, end='')
                print()
                print('ndcg     : @1 @2 @5 @10 @20 @50 @100 \n', end='')
                for i in ndcg:
                    print('%.4f\t' % i, end='')
                print('\n') 

    # torch.save(model, './ssml/model/net_model_v2.pkl')
    metrics = pd.DataFrame([epoch_list, train_loss_list, test_loss_list,
                            R_at1_list, R_at2_list, R_at5_list, R_at10_list, R_at20_list, R_at50_list, R_at100_list,
                            N_at1_list, N_at2_list, N_at5_list, N_at10_list, N_at20_list, N_at50_list, N_at100_list,
                            P_at1_list, P_at2_list, P_at5_list, P_at10_list, P_at20_list, P_at50_list, P_at100_list]).transpose()
    
    metrics.columns = ['epoch', 'train_loss', 'test_loss',
                    'R@1', 'R@2', 'R@5', 'R@10', 'R@20', 'R@50', 'R@100',
                    'N@1', 'N@2', 'N@5', 'N@10', 'N@p20', 'N@50', 'N@100',
                    'P@1', 'P@2', 'P@5', 'P@10', 'P@20', 'P@50', 'P@100']

    date = time.strftime('%Y-%m-%d', time.localtime(time.time()))
    metrics.to_csv('./{}/result/{}_{}_bs{}_lr{}_ep{}_{}.tsv'.format(args.model_name, args.model_name, args.dataset, args.batch_size, args.lr, args.n_epochs, date), sep='\t', index=False, encoding='utf-8')


def numpy2dataframe(numpy_data):
    userID = numpy_data[:, 0].tolist()
    itemID = numpy_data[:, 1].tolist()
    label = numpy_data[:, 2].tolist()
    df_data = pd.DataFrame([userID, itemID, label]).transpose()
    df_data.columns = ['userID', 'itemID', 'label']
    return df_data

def ctr_eval(model, criterion, data_loader, device):
    all_loss = 0.0
    # all_auc= 0.0
    # all_f1= 0.0
    for i, (user_ids, item_ids, labels) in enumerate(data_loader):
        user_ids, item_ids, labels = user_ids.to(device), item_ids.to(device), labels.to(device)
        if len(user_ids) == args.batch_size:  
            outputs = model(user_ids, item_ids)  
            loss = criterion(outputs, labels)
                
            # labels = labels.cpu().detach().numpy()
            # outputs = outputs.cpu().detach().numpy()
            # auc = roc_auc_score(y_true=labels, y_score=outputs)
            # predictions = np.array([1 if i >= 0.5 else 0 for i in outputs])
            # f1 = f1_score(y_true=labels, y_pred=predictions)

            all_loss += loss.item()
            # all_auc += auc
            # all_f1 += f1
    return all_loss / len(data_loader)


def topk_eval(model, user_list, train_record, test_record, item_set, k_list, batch_size, device):
    precision_list = {k: [] for k in k_list}
    recall_list = {k: [] for k in k_list}
    ndcg_list = {k: [] for k in k_list}

    for user in user_list:
        test_item_list = list(item_set - train_record[user])
        item_score_map = dict()
        start = 0

        while start + batch_size <= len(test_item_list):
            user_batch = [user] * batch_size
            user_batch = torch.from_numpy(np.array(user_batch)).to(device)
            item_batch = test_item_list[start: start + batch_size]
            items = item_batch
            item_batch = torch.from_numpy(np.array(item_batch)).to(device)

            scores = model(user_batch, item_batch)
            scores = scores.cpu().detach().numpy()

            for item, score in zip(items, scores):
                item_score_map[item] = score
            start += batch_size

        # padding the last incomplete minibatch if exists
        if start < len(test_item_list):
            user_batch = [user] * batch_size
            user_batch = torch.from_numpy(np.array(user_batch)).to(device)
            item_batch = test_item_list[start:] + [test_item_list[-1]] * (batch_size - len(test_item_list) + start)
            items = item_batch
            
            item_batch = torch.from_numpy(np.array(item_batch)).to(device)
            scores = model(user_batch, item_batch)

            scores = scores.cpu().detach().numpy()
            for item, score in zip(items, scores):
                item_score_map[item] = score

        item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
        item_sorted = [i[0] for i in item_score_pair_sorted]

        for k in k_list:
            hit_num = len(set(item_sorted[:k]) & test_record[user])
            precision_list[k].append(hit_num / k)
            recall_list[k].append(hit_num / len(test_record[user]))

            ndcg = nDCG(item_sorted[:k], list(test_record[user]))
            ndcg_list[k].append(ndcg)
        
    precision = [np.mean(precision_list[k]) for k in k_list]
    recall = [np.mean(recall_list[k]) for k in k_list]
    ndcg = [np.mean(ndcg_list[k]) for k in k_list]
    return precision, recall, ndcg


def IDCG(n):
    idcg = 0
    for i in range(n):
        idcg += 1 / math.log(i + 2, 2)
    return idcg

def nDCG(ranked_list, ground_truth):
    idcg = IDCG(len(ground_truth))

    dcg = 0.0
    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id not in ground_truth:
            continue
        rank = i + 1
        dcg += 1 / math.log(rank + 1, 2)
    
    ndcg = float(dcg / idcg) if idcg != 0 else 0
    return ndcg


def topk_settings(train_data, test_data, num_item):
    user_num = 100
    k_list = [1, 2, 5, 10, 20, 50, 100]
    train_record = get_user_record(train_data, True)
    test_record = get_user_record(test_data, False)
    user_list = list(set(train_record.keys()) & set(test_record.keys()))
    if len(user_list) > user_num:
        user_list = np.random.choice(user_list, size=user_num, replace=False)
    item_set = set(list(range(num_item)))

    user_click_item_pos = defaultdict(list) 
    for k, v in train_record.items():
        if len(v) >= args.click_sequence_size:
            values = random.sample(list(v), k=args.click_sequence_size)
        else:
            values = random.choices(list(v), k=args.click_sequence_size)
        user_click_item_pos[k] = values
    return user_list, train_record, test_record, item_set, k_list, user_click_item_pos


def get_user_record(data, is_train):
    user_history_dict = dict()
    for interaction in data:
        user = int(interaction[0])
        item = int(interaction[1])
        label = int(interaction[2])
        if is_train or label == 1:
            if user not in user_history_dict:
                user_history_dict[user] = set()
            user_history_dict[user].add(item)
    return user_history_dict
    

if __name__ == '__main__':
    main()
