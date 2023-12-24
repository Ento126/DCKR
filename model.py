import sys
import random
import numpy as np

import torch
from torch import nn

from ktMaxDiffuCl.aggregator import Aggregator
from ktMaxDiffuCl.context import Context
from ktMaxDiffuCl.gaussian_diffusion import *
from ktMaxDiffuCl.transform import Transform


def inverse_diffusion_fn(model_mlp, x_0, ent_emb, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps, device):
    # print('x_0.shape', x_0)  # x_0.shape torch.Size([128])
    x_0 = ent_emb(x_0)  # torch.Size([128, 16])

    batch_size = x_0.shape[0]
    t = torch.randint(0, n_steps, size=(batch_size//2,)).to(device=device)
    t = torch.cat([t, n_steps-1-t], dim=0)   # [batch_size]
    t = t.unsqueeze(-1)  # [batch_size, 1]  # t （bz,1）
    
    # (alpha_bar_t)
    a = alphas_bar_sqrt[t]  # a.shape  torch.Size([128, 1])
    # eps (1-alpha_bar_t)
    aml = one_minus_alphas_bar_sqrt[t]  # aml.shape torch.Size([128, 1])
    
    #  eps
    e = torch.randn_like(x_0).to(device=device)  # torch.Size([128, 16])

    a = a.to(device)
    aml = aml.to(device)

    x = x_0 * a + e * aml   # torch.Size([128, 16])
    output = model_mlp(x, t.squeeze(-1))  # torch.Size([128, 16])
    return output, e


class MLPDiffusion(nn.Module):
    def __init__(self, n_steps, num_units=128):
        super(MLPDiffusion,self).__init__()

        self.linears = nn.ModuleList([
                nn.Linear(16, num_units),
                nn.ReLU(),
                nn.Linear(num_units,num_units),
                nn.ReLU(),
                nn.Linear(num_units,num_units),
                nn.ReLU(),
                nn.Linear(num_units, 16), ])

        self.step_embeddings = nn.ModuleList([
                nn.Embedding(n_steps, num_units),
                nn.Embedding(n_steps, num_units),
                nn.Embedding(n_steps, num_units), ])

    def forward(self, x, t):
        for idx, embedding_layer in enumerate(self.step_embeddings):
            t_embedding = embedding_layer(t)  # t embedding

            x = self.linears[2 * idx](x) 
            x += t_embedding
            x = self.linears[2 * idx+1](x)
        x = self.linears[-1](x)
        return x


class ktMaxDiffuCl(torch.nn.Module):
    def __init__(self, usr_emb, ent_emb, rel_emb, adj_entity, adj_relation, user_click_item_pos, args, device):
        super(ktMaxDiffuCl, self).__init__()
        # n_user, n_item, n_entity, n_relation
        # self.num_user = n_user  # 1872
        # self.num_ent = n_entity    # 9366
        # self.num_rel = n_relation    # 60
        # self.usr = torch.nn.Embedding(self.num_user, args.dim)
        # self.ent = torch.nn.Embedding(self.num_ent, args.dim)
        # self.rel = torch.nn.Embedding(self.num_rel, args.dim)
        self.usr = usr_emb
        self.ent = ent_emb
        self.rel = rel_emb
        # self.kg = kg
        # self.adj_entity = adj_entity
        # self.adj_relation = adj_relation
        self.adj_ent = adj_entity
        self.adj_rel = adj_relation
        self.user_click_item_sequence = user_click_item_pos
        self.device = device

        self.n_iter = args.n_iter  # 1
        self.batch_size = args.batch_size  # 128
        self.click_sequence_size = args.click_sequence_size
        self.dim = args.dim  # 16
        self.n_neighbor = args.neighbor_sample_size  # 8
        self.num_steps = args.num_steps

        self.input_size = args.rnn_input_size
        self.hidden_size = args.rnn_hidden_size
        self.num_layers = args.rnn_num_layers

        self.pad_size = args.click_sequence_size
        self.dropout = 0.2

        # function
        self.aggregator = Aggregator(self.batch_size, self.dim, args.aggregator)

        # bidirectional设为False即得到单向循环神经网络
        self.gru = nn.GRU(input_size=self.dim,
                               hidden_size=self.hidden_size,
                               num_layers=self.num_layers,
                               batch_first=False,
                               bidirectional=True)

        self.mha = nn.MultiheadAttention(self.dim, num_heads = 1)

        self.relu = nn.ReLU()

        self.context = Context(self.dim, self.hidden_size, self.num_layers)

        self.transform = Transform(self.dim, self.pad_size, self.dropout, self.device)

        self.model_mlp = MLPDiffusion(args.num_steps).to(device=device) 
        optimizer_mlp = torch.optim.Adam(self.model_mlp.parameters(),lr=1e-3)
        criterion1 = torch.nn.MSELoss()

        self.criterion = torch.nn.BCELoss()

        betas = torch.linspace(-6,6, args.num_steps)
        betas = torch.sigmoid(betas)*(0.5e-2 - 1e-5)+1e-5  # torch.Size([100])

        #alpha、alpha_prod、alpha_prod_previous、alpha_bar_sqrt
        alphas = 1-betas
        alphas_prod = torch.cumprod(alphas, 0)
        alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]], 0)
        self.alphas_bar_sqrt = torch.sqrt(alphas_prod)
        one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)
        assert alphas.shape==alphas_prod.shape==alphas_prod_p.shape==\
        self.alphas_bar_sqrt.shape==one_minus_alphas_bar_log.shape\
        ==self.one_minus_alphas_bar_sqrt.shape
        print("all the same shape",betas.shape)


    def _aggregate(self, user_embeddings, entities, relations, ):
        """
        Make item embeddings by aggregating neighbor vectors
        """
        entity_vectors = []
        entity_vectors = [self.ent(entity.to(device=self.device)) for entity in entities]
        relation_vectors = [self.rel(relation.to(device=self.device)) for relation in relations]

        for i in range(self.n_iter):
            if i == self.n_iter - 1:
                act = torch.tanh
            else:
                act = torch.sigmoid

            entity_vectors_next_iter = []
            for hop in range(self.n_iter - i):
                # print(entity_vectors[hop].shape)
                # print('测试self_vectors---------------')
                vector = self.aggregator(
                    self_vectors = entity_vectors[hop],
                    neighbor_vectors = entity_vectors[hop + 1].view((self.click_sequence_size, -1, self.n_neighbor, self.dim)),
                    neighbor_relations = relation_vectors[hop].view((self.click_sequence_size, -1, self.n_neighbor, self.dim)),
                    user_embeddings = user_embeddings,
                    act=act)
                entity_vectors_next_iter.append(vector)

            entity_vectors = entity_vectors_next_iter
        return entity_vectors[0].view((self.click_sequence_size, self.dim))
    
    def _get_neighbors(self, v):
        """
        v is batch sized indices for items  128
        v: [batch_size, 1]
        """
        entities = [v]  # 128 list
        relations = []
        for h in range(self.n_iter):
            neighbor_entities = torch.LongTensor(self.adj_ent[entities[h]].cpu()).view((self.click_sequence_size, -1)).to(self.device)
            neighbor_relations = torch.LongTensor(self.adj_rel[entities[h]].cpu()).view((self.click_sequence_size, -1)).to(self.device)
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)
        return entities, relations
    

    def forward_pretrain(self, u, v, labels):
            
        user_diff, e = inverse_diffusion_fn(self.model_mlp, u, self.usr, self.alphas_bar_sqrt, self.one_minus_alphas_bar_sqrt, self.num_steps, self.device)
        diffu_loss = (e - user_diff).square().mean()

        """KGCN model"""
        user_embeddings = self.usr(u.view((-1, 1))).squeeze(dim=1)

        batch_size = u.size(0)
        u = u.type(torch.int64)

        u_aggregate_item_embeddings = torch.empty(batch_size, self.click_sequence_size, self.dim, dtype=torch.float32)
        for i in range(batch_size):
            ui = u[i].item()
            ui_click_item = list(self.user_click_item_sequence[ui])
            if len(ui_click_item) == 0:
                ui = int(1515)
                ui_click_item = list(self.user_click_item_sequence[ui])

            ui_click_item = torch.from_numpy(np.array(ui_click_item)).view((-1, 1)).long()   # torch.Size([click_sequence_size, 1])
            entities, relations = self._get_neighbors(ui_click_item)
            item_embeddings = self._aggregate(user_diff, entities, relations)  # torch.Size([click_sequence_size, dim]) 
            u_aggregate_item_embeddings[i] = item_embeddings

        u_aggregate_item_embeddings = self.relu(u_aggregate_item_embeddings)
        # device
        u_aggregate_item_embeddings = u_aggregate_item_embeddings.to(device=self.device)
        u_aggregate_item_embeddings = self.relu(u_aggregate_item_embeddings)
        # feat = self.transform(x=gru_outputs)
        trans_out = self.transform(x=u_aggregate_item_embeddings)
        trans_sum_out = torch.sum(trans_out, dim=1)  # transformer feature
        trans_sum_out = torch.relu(trans_sum_out)
        trans_max_pool = trans_out.max(dim=1).values  # transformer max
        trans_max_pool = torch.relu(trans_max_pool)
        trans_fusion = torch.cat((trans_sum_out, trans_max_pool), dim=1)  # transformer feature + transformer max

        fc1 = nn.Linear(in_features = trans_fusion.size(1), out_features = 64).to(self.device)
        fc2 = nn.Linear(in_features = 64, out_features = self.dim).to(self.device)
        feat = self.relu(fc1(trans_fusion))  # (batch_size, dim)
        feat = self.relu(fc2(feat))  # (batch_size, dim)

        item_embeddings_v = self.ent(v).squeeze(dim=1)
        scores = (feat * item_embeddings_v).sum(dim=1)
        outs = torch.sigmoid(scores)
        Rec_loss = self.criterion(outs, labels)
        user_level_embedding = u_aggregate_item_embeddings.mean(dim=1)
        user_level_embedding = self.relu(user_level_embedding)
        sim_score_users = torch.mul(user_level_embedding, user_embeddings).sum(dim=1)  
        sim_score_users = torch.sigmoid(sim_score_users)
        pre_users_label = torch.ones_like(sim_score_users, dtype=torch.float32)
        ssl_loss_users = self.criterion(sim_score_users, pre_users_label)
        loss = diffu_loss + Rec_loss + ssl_loss_users   # torch.Size([batch_size]])
        return loss
    
    def forward(self, u, v):
        """KGCN model"""
        user_embeddings = self.usr(u.view((-1, 1))).squeeze(dim=1)

        batch_size = u.size(0)
        u = u.type(torch.int64)

        u_aggregate_item_embeddings = torch.empty(batch_size, self.click_sequence_size, self.dim, dtype=torch.float32)
        for i in range(batch_size):
            ui = u[i].item()
            ui_click_item = list(self.user_click_item_sequence[ui])
            if len(ui_click_item) == 0:
                ui = int(1515)
                ui_click_item = list(self.user_click_item_sequence[ui])

            ui_click_item = torch.from_numpy(np.array(ui_click_item)).view((-1, 1)).long()   # torch.Size([click_sequence_size, 1])
            entities, relations = self._get_neighbors(ui_click_item)  # 获取v的实体和关系
            item_embeddings = self._aggregate(user_embeddings, entities, relations)  # torch.Size([click_sequence_size, dim])          
            u_aggregate_item_embeddings[i] = item_embeddings

        u_aggregate_item_embeddings = self.relu(u_aggregate_item_embeddings)
        # device
        u_aggregate_item_embeddings = u_aggregate_item_embeddings.to(device=self.device)
        u_aggregate_item_embeddings = self.relu(u_aggregate_item_embeddings)
        trans_out = self.transform(x=u_aggregate_item_embeddings)

        trans_sum_out = torch.sum(trans_out, dim=1)  # transformer feature
        trans_sum_out = torch.relu(trans_sum_out)
        trans_max_pool = trans_out.max(dim=1).values  # transformer max
        trans_max_pool = torch.relu(trans_max_pool)
        trans_fusion = torch.cat((trans_sum_out, trans_max_pool), dim=1)  # transformer feature + transformer max
        fc1 = nn.Linear(in_features = trans_fusion.size(1), out_features = 64).to(self.device)
        fc2 = nn.Linear(in_features = 64, out_features = self.dim).to(self.device)
        feat = self.relu(fc1(trans_fusion))  # (batch_size, dim)
        feat = self.relu(fc2(feat))  # (batch_size, dim)
        item_embeddings_v = self.ent(v).squeeze(dim=1)
        scores = (feat * item_embeddings_v).sum(dim=1)  # 计算相似结果 torch.Size([batch_size])
        outs = torch.sigmoid(scores)
        return outs  # torch.Size([batch_size]])
