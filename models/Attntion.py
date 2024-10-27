import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import math


class Attention(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
    """

    def __init__(self, hidden, use_temporal_attn = True):
 
        super(Attention, self).__init__()

        self.hidden = hidden

        self.use_temporal_attn = use_temporal_attn

        self.ln = nn.LayerNorm(hidden*3,)

        self.fc1 = nn.Linear(hidden, hidden, bias=True,)

        self.fc2 = nn.Linear(hidden, hidden, bias=True,)

        self.fc3 = nn.Conv1d(hidden, hidden, kernel_size=7, stride=1, padding=3, groups=hidden)

        self.fc4 = nn.Linear(hidden, hidden,)
        self.fc5 = nn.Linear(hidden*2, hidden,)
        self.pool1 = nn.AdaptiveAvgPool1d(8)
        self.pool2 = nn.AdaptiveAvgPool1d(8)
        self.gru1 = nn.GRU(hidden, hidden, 1)
        self.gru2 = nn.GRU(hidden, hidden, 1)
        self.gru3 = nn.GRU(hidden, hidden, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def TemAttn(self,q,k,v):

        q = self.pool1(q)
        k = self.pool2(k)
        score = torch.matmul(q,k.transpose(-1,-2))
        
        score = self.softmax(score)
        x = torch.matmul(score,v)
        return x
    
    def SpaAttn(self,q,k,v):
        q = self.gru1(q)[0]
        k = self.gru2(k)[0]
        v = self.gru3(v)[0]
        score = torch.matmul(q,k.transpose(-1,-2))
        score = self.tanh(score)
        b,p1,p2 = score.shape
        score = score.view(b,self.hidden,p1*p2//self.hidden)
        score = self.fc3(score)
        score = score.view(b,p1,p2)
        score = self.softmax(score)
        x = torch.matmul(score,v)
        return x

    def forward(self, q,k,v):
        x_origin = q.clone()
        x_0 = self.SpaAttn(q,k,v)
        if self.use_temporal_attn:
            x_1 = self.TemAttn(q,k,v)
        else:
            x_1 = x_0

        x = torch.cat((x_0,x_1),dim=-1)
        x = self.fc5(x) + x_origin

        return x
