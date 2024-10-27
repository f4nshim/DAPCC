import torch.nn as nn
import torch
import random

class Aggregation(nn.Module):
    def __init__(self, embed_dim,k):
        super(Aggregation, self).__init__()
        self.k = k
        self.embed_dim  = embed_dim

        self.scorecalrefer1 = nn.Linear(embed_dim, embed_dim)

        self.scorecaltemp1 = nn.Linear(embed_dim, embed_dim)
        
        self.mlp = nn.Linear(embed_dim,embed_dim)

        self.norm = nn.BatchNorm1d(k)
        
        self.norm1 = nn.BatchNorm1d(k)
    
    def forward(self,temp,refer):
        b, seq_num, _  = temp.shape
        if len(refer.shape) == 3:
            flat_refer = refer.flatten(0,1)
        else:
            flat_refer = refer
        index = torch.LongTensor(random.sample(range(flat_refer.shape[0]), 1 * seq_num))
        mask = torch.ones(flat_refer.shape[0], dtype=torch.bool)
        mask[index] = False
        deeper = flat_refer[~mask]

        flat_refer = flat_refer[mask]

        deeper_score = self.scorecalrefer1(deeper)

        temp_score = self.scorecaltemp1(temp)
        
        score = torch.matmul(deeper_score,temp_score.transpose(-2,-1)/(self.embed_dim**0.5))

        top_score, index = torch.topk(score.view(b,1*seq_num,seq_num//self.k,self.k).sum(-1),k=self.k,dim=1)
        top_score = top_score.squeeze(1)
        index = index.squeeze(1).flatten(1,2)
        top_score = self.norm(top_score).flatten(1,2)
        deeper = deeper[index]

        return temp, flat_refer, deeper
    


    






    