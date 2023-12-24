import sys
import torch
import torch.nn as nn
import torch.nn.functional as F 
import math


class Context(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, n_layers):
        
        super(Context, self).__init__()
        self.dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.rnn = nn.GRU(input_size=self.dim,
                        hidden_size=self.hidden_dim,
                        num_layers=self.n_layers,
                        batch_first=False,
                        bidirectional=True)
        
        self.mha = nn.MultiheadAttention(self.dim, num_heads = 1)
       
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()


    def forward(self, x):
        # [seq_len, batch, embedding_dim]        
        # output:[seq_len, batch, hidden_dim*2]
        # hidden/cell:[n_layers*2, batch, hidden_dim]


        gru_outputs, _ = self.rnn(x)   # outputs (batch_size, seq_len, 2*num_hiddens)
        gru_outputs = self.relu(gru_outputs)
        # print(gru_outputs.shape)
        # print('gru_outputs-------------------')

        # self-Attention
        att_x = gru_outputs  # x (batch_size, seq_len, num_hiddens)
        attn_output, attn_output_weights = self.mha(att_x, att_x, att_x)   # (batch_size, seq_len, num_hiddens)
        attn_output = self.relu(attn_output)
        # print(attn_output.shape)
        # print('attn_output-------------------')

        merge_x_attn_out = torch.cat([gru_outputs, attn_output], dim=2)  # (batch_size, seq_len, 2*dim)
        # print(merge_x_attn_out.shape)
        # print('merge_x_attn_out-------------------')
        # sys.exit(0)

        return merge_x_attn_out


    def attention_net(self, x, query, mask=None): 
        d_k = query.size(-1) 
        # query:[batch, seq_len, hidden_dim*2], x.t:[batch, hidden_dim*2, seq_len]
#         print("query: ", query.shape, x.transpose(1, 2).shape)  # torch.Size([128, 38, 128]) torch.Size([128, 128, 38])
        # scores: [batch, seq_len, seq_len]
        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)
#         print("score: ", scores.shape)  # torch.Size([128, 38, 38])

        alpha_n = F.softmax(scores, dim=-1)
#       print("alpha_n: ", alpha_n.shape)    # torch.Size([128, 38, 38]
        # [batch, seq_len, seq_len]·[batch,seq_len, hidden_dim*2] = [batch,seq_len,hidden_dim*2] -> [batch, hidden_dim*2]
        context = torch.matmul(alpha_n, x).sum(1)
        
        return context, alpha_n
