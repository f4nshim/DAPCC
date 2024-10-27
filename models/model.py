import torch
import torch.nn as nn
from models.LOAEM import LOAEM
import os


class Net(nn.Module):
    def __init__(self,args):
        super().__init__()
        if args.weights != "":
            assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
            self.model = torch.load(args.weights, map_location=args.device[0],weights_only=False)
            print('load dict:',args.weights)
        else:
            self.model = LOAEM(
                sequence_size=1024,
                dropout_rate= args.droprate,
                hidden=args.hid,
                topk=args.topk, 
                num_layer=args.nlayer,
                use_aggre=args.use_aggre,
                use_temporal_attn=args.use_temporal_attn,
                use_distilling=args.use_distilling,
                )
        self.model.to(args.device[0])


    def forward(self,temp,refer):
        occu = self.model(temp,refer)
        return occu

