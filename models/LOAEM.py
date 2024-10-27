import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.nn.modules import Module
from torch.nn.modules.transformer import _get_clones
import sys

sys.path.append("../")
from flow_compression.models.Attntion import Attention
from models.aggregation import Aggregation
import torch.nn.functional as F


class LeTransformerEncoderLayer(Module):

    def __init__(
            self, 
            d_model, 
            dropout = 0.1, 
            layer_norm_eps = 1e-5,
            use_temporal_attn = True,
            use_distilling = True,
            ):

        super(LeTransformerEncoderLayer, self).__init__()

        self.attn = Attention(d_model, use_temporal_attn,)
        # Implementation of Conv Feedforward model

        self.dropout = nn.Dropout(dropout)

        self.distill1 = nn.Linear(d_model//2, d_model//4,)
        
        self.distill2 = nn.Linear(d_model//4, d_model,)
        
        self.linear2 = nn.Linear(d_model//2, d_model,)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps,)
        
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps,)
        
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps,)
        
        self.norm4 = nn.LayerNorm(d_model,  eps=layer_norm_eps,)

        self.dropout1 = nn.Dropout(dropout,)
        
        self.dropout2 = nn.Dropout(dropout,)

        self.sa_distilling = nn.Sequential(
            nn.Linear(d_model,d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2,d_model//4),
            nn.ReLU(),
            nn.Linear(d_model//4,d_model),
            nn.Dropout(dropout),
            )

        self.ca_distilling = nn.Sequential(
            nn.Linear(d_model,d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2,d_model//4),
            nn.ReLU(),
            nn.Linear(d_model//4,d_model),
            nn.Dropout(dropout),
            )
        
        self.use_distilling = use_distilling

    def forward(self, src, attn):

        x = src

        x = self.norm1(self._ca_block(x, attn))
        if self.use_distilling:
            x = self.norm2(x + self.ca_distilling(x))

        x = self.norm3(self._sa_block(x))
        if self.use_distilling:
            x = self.norm4(x + self.sa_distilling(x))

        return x

    # self-attention block
    def _ca_block(self, x, attn):
        
        x = self.attn(x, attn, x)

        return self.dropout1(x)
    
    def _sa_block(self, x):
        
        x = self.attn(x, x, x)

        return self.dropout2(x)


class OctFormerEncoder(Module):

    def __init__(
            self, 
            encoder_layer, 
            num_layers, 
            d_model=128,
            topk = 8, 
            use_aggre = True,
            ):
        
        super(OctFormerEncoder, self).__init__()

        self.num_layers = num_layers
        self.use_aggre = use_aggre

        self.layers = _get_clones(encoder_layer, num_layers,)

        self.mlp = nn.Linear(d_model, d_model,)

        self.pos_block = nn.ModuleList(Aggregation(embed_dim=d_model, k = topk,) for i in range(num_layers))
        
    def forward(self, src, refer):

        output = src

        for index, mod in enumerate(self.layers):
 
            if self.use_aggre:
                output,refer,deep = self.pos_block[index//3](output, refer)
                output = mod(output, attn = deep)
            else:
                deep = self.mlp(refer[random.randint(0, refer.shape[0]-1)]) 
                # randomly select a sequence in reference frame as node if not have aggregation embedding
                output = mod(output,attn = deep)

        return output


class LOAEM(nn.Module):

    def __init__(
            self, 
            sequence_size, 
            dropout_rate, 
            hidden=128, 
            topk=8, 
            num_layer=6, 
            use_aggre = True, 
            use_temporal_attn = True, 
            use_distilling = True
            ):
        
        print(
            "LOAEM: sequence_size: {}, topk: {}, num_layer: {}, hidden: {}, dropout: {}, use_aggre: {}, use_temporal_attn: {}, use_distilling: {}".format(
                sequence_size,
                topk,
                num_layer,
                hidden,
                dropout_rate,
                use_aggre,
                use_temporal_attn,
                use_distilling
            ))
        
        super(LOAEM, self).__init__()

        self.sequence_size = sequence_size

        self.embed_current = nn.Linear(in_features=6, out_features=hidden,)
        
        self.embed_refer = nn.Linear(in_features=6, out_features=hidden,)
        
        self.encoder_layer = LeTransformerEncoderLayer(
            d_model = hidden, 
            dropout = dropout_rate, 
            layer_norm_eps = 1e-5,
            use_temporal_attn = True,
            use_distilling = True,
            )

        self.transformerEncoder = OctFormerEncoder(
            self.encoder_layer, 
            num_layers=num_layer, 
            d_model=hidden, 
            topk = topk,
            use_aggre=use_aggre,
            )
        
        self.MLP1 = nn.Linear(in_features=hidden, out_features=hidden,)
        
        self.MLP2 = nn.Linear(in_features=hidden, out_features=256,)
        
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, features,refer):
        out = self.embed_current(features)
        refer = self.embed_refer(refer)

        out = self.transformerEncoder(out,refer)
        out = self.MLP1(out)
        out = self.dropout(out)
        out = self.MLP2(out[:,:1024])
        # print(out[2].max(dim=-1)[1][:20])
        return out


if __name__ == '__main__':
    model = LOAEM(
        sequence_size = 128, 
        dropout_rate = 0.5, 
        hidden = 256,
        topk = 8, 
        num_layer = 3,
        use_aggre = True,
        use_temporal_attn = True,
        use_distilling = True,
        )
    currrent = torch.zeros((64, 16, 6))  # batch_size =  64, sequence = 16, dimention=6
    refer = torch.zeros((50,16,6))
    print(model(currrent, refer))
    # torch.save(model,"1.pth")
