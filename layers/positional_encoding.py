import math
import torch
import torch.nn as nn


class PositionalEncodingSinCos(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=250):
        super(PositionalEncodingSinCos, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class PositionalEncodingLUT(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=250):
        super(PositionalEncodingLUT, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len, dtype=torch.long).unsqueeze(1)
        self.register_buffer('position', position)

        self.pos_embed = nn.Embedding(max_len, d_model)
        self.d_model = d_model
        self._init_embeddings()

    def _init_embeddings(self):
        nn.init.kaiming_normal_(self.pos_embed.weight, mode="fan_in")

    def forward(self, x):
        #print('self.d_model: ',self.d_model)
        '''self.d_model:  256'''
        #print('x.shape: ',x.shape)
        '''x.shape:  torch.Size([60, 10, 256])'''
        #print('self.position: ',self.position)
        '''from 0 to 59'''
        #print('self.position.shape: ',self.position.shape)
        '''self.position.shape:  torch.Size([62, 1])'''
        pos = self.position[:x.size(0)]
        #print('pos.shape: ',pos.shape)
        '''pos.shape:  torch.Size([60, 1])'''
        x = x + self.pos_embed(pos)
        #print('x: ',x)
        #print('x2.shape: ',x.shape)
        '''x2.shape:  torch.Size([60, 10, 256]'''
        return self.dropout(x)
