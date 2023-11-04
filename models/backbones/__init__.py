import torch.nn as nn
import torch
from .slowfast import *
# from utils import load_pretrain


def model_entry(config):
    return globals()[config['arch']](**config['kwargs'])


class AVA_backbone(nn.Module):
    def __init__(self, config):
        super(AVA_backbone, self).__init__()
        
        self.config = config
        self.module = model_entry(config.model.backbone)
        print(config.get('pretrain', None))
        if config.get('pretrain', None) is not None:
            load_pretrain(config.pretrain, self.module)
                
        if not config.get('learnable', True):
            self.module.requires_grad_(False)

    # data: clips
    # returns: features
    def forward(self, data):
        # inputs = data['clips']
        inputs = data
        inputs = inputs.cuda()
        features = self.module(inputs)
        return features
    




def load_pretrain(pretrain_opt, net):
    checkpoint = torch.load(pretrain_opt.path, map_location=lambda storage, loc: storage.cuda())
    if pretrain_opt.get('state_dict_key', None) is not None:
        checkpoint = checkpoint[pretrain_opt.state_dict_key]

    if pretrain_opt.get('delete_prefix', None):
        keys = set(checkpoint.keys())
        for k in keys:
            if k.startswith(pretrain_opt.delete_prefix):
                checkpoint.pop(k)
    if pretrain_opt.get('replace_prefix', None) is not None:
        keys = set(checkpoint.keys())
        for k in keys:
            if k.startswith(pretrain_opt.replace_prefix):
                new_k = pretrain_opt.get('replace_to', '') + k[len(pretrain_opt.replace_prefix):]
                checkpoint[new_k] = checkpoint.pop(k)
    net.load_state_dict(checkpoint, strict=False)
