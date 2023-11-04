import torch.nn as nn
import math, pdb
import torch.nn.functional as F
import torch
from torch.nn.parameter import Parameter
from models.nonlocal_helper import Nonlocal
from models.blocks import *
from models.backbones import swin_transformer
import modules.utils as lutils

import yaml
from easydict import EasyDict
from .backbones import AVA_backbone

from functools import reduce, lru_cache
from operator import mul
from einops import rearrange
from .resnetFPN_DINOV2 import YOLOPAFPN3DDINO


logger = lutils.get_logger(__name__)

### Download weights from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
def conv3x3(in_planes, out_planes, stride=1, padding=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1, 3, 3), stride=(1, stride, stride),
                     padding=(0,padding,padding), bias=bias)


def conv1x1(in_channel, out_channel):
    return nn.Conv3d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False)



# swinmodel=swin_transformer.SwinTransformer3D()



#good


class ResNetFPN(nn.Module):

    def __init__(self, block, args, pretrained=None,
                 pretrained2d=True,
                 patch_size=(2, 4, 4),
        embed_dim=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=(8, 7, 7),
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
                 drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.4,
        patch_norm=True,
                 norm_layer=nn.LayerNorm,
                 frozen_stages=4,
                 use_checkpoint=False):
        
        self.inplanes = 64
        super(ResNetFPN, self).__init__()
    
    
        if args.MODEL_TYPE.startswith('SlowFast'):
            with open('configs/ROAD.yml') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            opt = EasyDict(config)
            print(opt)
            self.inplanes = 64
            super(ResNetFPN, self).__init__()
            self.backbone = AVA_backbone(opt)
    


        depths=[2, 2, 18, 2]
        num_heads=[6, 12, 24, 48]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule


        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.frozen_stages = frozen_stages
        self.window_size = window_size
        self.patch_size = patch_size



        self.MODEL_TYPE = args.MODEL_TYPE
        num_blocks = args.model_perms
        non_local_inds = args.non_local_inds
        model_3d_layers = args.model_3d_layers
        self.num_blocks = num_blocks
        self.non_local_inds = non_local_inds
        self.model_3d_layers = model_3d_layers
        self.patch_norm = True
        self.patch_size=patch_size
        self.window_size=window_size
        self.num_layers=4
        self.num_features = int(embed_dim * 2**(self.num_layers-1))
        self.patch_embed=swin_transformer.PatchEmbed3D(patch_size=self.patch_size, in_chans=3, embed_dim=embed_dim,
            norm_layer=nn.LayerNorm if norm_layer else None)#embed_dim 4 56 56
        
        self.pos_drop = nn.Dropout(p=0.0)
        # self.layers = nn.ModuleList()
        
        self.swinaclayer1=swin_transformer.BasicLayer(dim=int(embed_dim*2**0),depth=depths[0],num_heads=num_heads[0],window_size=self.window_size,
                                                      qkv_bias=True,drop_path=dpr[sum(depths[:0]):sum(depths[:1])],downsample=swin_transformer.PatchMerging if 0<self.num_layers-1 else None)#384 4 28 28
        
        # self.patchmerge1=swin_transformer.PatchMerging(dim=192)#384 4 28 28
        
        self.swinaclayer2=swin_transformer.BasicLayer(dim=embed_dim*2**1,depth=2,num_heads=12,window_size=(8,7,7),
                                                      qkv_bias=True,drop_path=dpr[sum(depths[:1]):sum(depths[:2])],downsample=swin_transformer.PatchMerging if 1<self.num_layers-1 else None)
        
        # self.patchmerge2=swin_transformer.PatchMerging(dim=192*2)#768 4 14 14
        
        self.swinaclayer3=swin_transformer.BasicLayer(dim=embed_dim*2**2,depth=18,num_heads=24,window_size=(8,7,7),
                                                      qkv_bias=True,drop_path=dpr[sum(depths[:2]):sum(depths[:3])],downsample=swin_transformer.PatchMerging if 2<self.num_layers-1 else None)#768 4 14 14
        
        # self.patchmerge3=swin_transformer.PatchMerging(dim=192*2**2)#1536 4 7 7
        
        self.swinaclayer4=swin_transformer.BasicLayer(dim=embed_dim*2**3,depth=2,num_heads=48,window_size=(8,7,7),
                                                      qkv_bias=True,drop_path=dpr[sum(depths[:3]):sum(depths[:3 + 1])],downsample=swin_transformer.PatchMerging if 3<self.num_layers-1 else None)#1536 4 7 7
        # self.layers.append(layer)
        

        





        self.norm3 = nn.LayerNorm(self.num_features)
        self.norm3.eval()
        for param in self.norm3.parameters():
                    param.requires_grad = False

        self._freeze_stages()
        
        
        self.upcov1=nn.ConvTranspose3d(384, 256, kernel_size=(2,2,2), stride=(2,2,2))
        self.upcov2=nn.ConvTranspose3d(768, 512, kernel_size=(2,2,2), stride=(2,2,2))
        self.upcov3=nn.ConvTranspose3d(1536, 1024, kernel_size=(2,2,2), stride=(2,2,2))
        self.upcov4=nn.ConvTranspose3d(1536, 2048, kernel_size=(2,1,1), stride=(2,1,1), padding=0)
        # self.swinlayer1=swinmodel.layers[0]
        # self.swinlayer2=swinmodel.layers[1]
        # self.swinlayer3=swinmodel.layers[2]
        # self.swinlayer4=swinmodel.layers[3]

        

        # self.conv1 = nn.Conv3d(3, 64, kernel_size=(1,7,7), stride=(1,2,2), padding=(0,3,3),
        #                        bias=False)
        # self.bn1 = nn.BatchNorm3d(64)
        # self.relu = nn.ReLU(inplace=True)
        # self.pool1 = nn.MaxPool3d(kernel_size=(
        #     1, 3, 3), stride=(1, 2, 2), padding=(0, 0, 0))
        self.layer_names = []

        self.layer1 = self._make_layer(
            block, 64, num_blocks[0], temp_kernals=model_3d_layers[0], nl_inds=non_local_inds[0])
        self.MODEL_TYPE = args.MODEL_TYPE
        
        # if args.model_subtype in ['C2D','RCN','CLSTM','RCLSTM','CGRU','RCGRU']:
        #     self.pool2 = None
        # else:
        #     self.pool2 = nn.MaxPool3d(kernel_size=(
        #         2, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        self.pool2=None
        self.layer2 = self._make_layer(
            block, 128, num_blocks[1], stride=2, temp_kernals=model_3d_layers[1], nl_inds=non_local_inds[1])
        self.layer3 = self._make_layer(
            block, 256, num_blocks[2], stride=2, temp_kernals=model_3d_layers[2], nl_inds=non_local_inds[2])
        self.layer4 = self._make_layer(
            block, 512, num_blocks[3], stride=2, temp_kernals=model_3d_layers[3], nl_inds=non_local_inds[3])

        #self.avgpool = nn.AvgPool2d(7, stride=1)
        #self.fc = nn.Linear(512 * block.expansion, num_classes)

        if self.MODEL_TYPE == 'SlowFast':
            self.conv6 = conv3x3(2304, 256, stride=2, padding=1)  # P6
            self.conv7 = conv3x3(256, 256, stride=2, padding=1)  # P7

            self.ego_lateral = conv3x3(512 * block.expansion,  256, stride=2, padding=0)
            self.avg_pool = nn.AdaptiveAvgPool3d((None, 1, 1))

            self.lateral_layer1 = conv1x1(2304, 256)
            self.lateral_layer2 = conv1x1(1152, 256)
            self.lateral_layer3 = conv1x1(576, 256)
            
            self.corr_layer1 = conv3x3(256, 256, stride=1, padding=1)  # P4
            self.corr_layer2 = conv3x3(256, 256, stride=1, padding=1)  # P4
            self.corr_layer3 = conv3x3(256, 256, stride=1, padding=1)  # P3
        else:
            self.conv6 = conv3x3(512 * block.expansion, 256, stride=2, padding=1)  # P6
            self.conv7 = conv3x3(256, 256, stride=2, padding=1)  # P7

            # self.ego_lateral = conv3x3(512 * block.expansion,  256, stride=2, padding=0)
            self.avg_pool = nn.AdaptiveAvgPool3d((None, 1, 1))

            self.lateral_layer1 = conv1x1(512 * block.expansion, 256)
            self.lateral_layer2 = conv1x1(256 * block.expansion, 256)
            self.lateral_layer3 = conv1x1(128 * block.expansion, 256)
            
            self.corr_layer1 = conv3x3(256, 256, stride=1, padding=1)  # P4
            self.corr_layer2 = conv3x3(256, 256, stride=1, padding=1)  # P4
            self.corr_layer3 = conv3x3(256, 256, stride=1, padding=1)  # P3


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_uniform_(m.weight, a=1)
                if hasattr(m.bias, 'data'):
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1:
            self.pos_drop.eval()
            for i in range(1, self.frozen_stages + 1):
                layer_name = 'swinaclayer{}'.format(i)
                layert = getattr(self, layer_name)
                m = layert
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False




    def _upsample(self, x, y):
        _, _, t, h, w = y.size()
        # print('spatial', x.shape, y.shape)
        x_upsampled = F.interpolate(x, [t, h, w], mode='nearest')

        return x_upsampled

    def _upsample_time(self, x):
        _,_,t, h, w = x.size()
        # print('time', x.shape)
        x_upsampled = F.interpolate(x, [t*2, h, w], mode='nearest')

        return x_upsampled

    def _make_layer(self, block, planes, num_blocks, stride=1, temp_kernals=[], nl_inds=[]):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=(1, stride, stride), bias=False),
                nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layer_names = []
        if 0 in temp_kernals:
            layers.append(block(self.inplanes, planes, stride,
                                temp_kernal=3, downsample=downsample))
        else:
            layers.append(block(self.inplanes, planes, stride,
                                temp_kernal=0, downsample=downsample))
        layer_names.append(0)
        self.inplanes = planes * block.expansion

        for i in range(1, num_blocks):
            if i in temp_kernals:
                layers.append(block(self.inplanes, planes, temp_kernal=3))
            else:
                layers.append(block(self.inplanes, planes, temp_kernal=1))
            layer_names.append(i)

            if i in nl_inds:
                nln = Nonlocal(
                    planes * block.expansion,
                    (planes * block.expansion) // 2,
                    [1, 2, 2],
                    instantiation='softmax',
                )
                layers.append(nln)

        self.layer_names.append(layer_names)

        return nn.Sequential(*layers)

    def forward(self, x):

        if self.MODEL_TYPE.startswith('SlowFast'):
            ff = self.backbone(x)
            c3 = ff[0]
            c4 = ff[1]
            c5 = ff[2]
            p5 = self.lateral_layer1(c5)
            p5_upsampled = self._upsample(p5, c4)
            p5 = self.corr_layer1(p5)
            p4 = self.lateral_layer2(c4)
            p4 = p5_upsampled + p4
            p4_upsampled = self._upsample(p4, c3)
            p4 = self.corr_layer2(p4)
            p3 = self.lateral_layer3(c3)
            p3 = p4_upsampled + p3
            p3 = self.corr_layer3(p3)
            p6 = self.conv6(c5)
            p7 = self.conv7(F.relu(p6))
            features = [p3, p4, p5, p6, p7]
            ego_feat = self.avg_pool(p7)
            if self.pool2 is not None:
                for i in range(len(features)):
                    features[i] = self._upsample_time(features[i])
                ego_feat = self._upsample_time(ego_feat)
        else:
            x=self.patch_embed(x)
            # print(x.shape)
            x=self.pos_drop(x)
            # print(x.shape)
            x=self.swinaclayer1(x)
            # print(x.shape)
            x1=x #192 4 56 56
            # x=self.patchmerge1(x)
            x=self.swinaclayer2(x)
            x2=x #384 4 28 28
            # x=self.patchmerge2(x)
            x=self.swinaclayer3(x)
            x3=x #768 4 14 14
            # x=self.patchmerge3(x)
            x=self.swinaclayer4(x)
            x = rearrange(x, 'n c d h w -> n d h w c')
            x = self.norm3(x)
            x = rearrange(x, 'n d h w c -> n c d h w')
            x4=x #1536 4 7 7

            #up can be pretrained

            u1=self.upcov1(x1)#256 4 56 56
            u2=self.upcov2(x2)#512 4 28 28
            u3=self.upcov3(x3)#1024 4 14 14
            u4=self.upcov4(x4)#2048 4 7 7
            
            
            
            
            # x = self.conv1(x)
            # x = self.bn1(x)
            # x = self.relu(x)
            # x = self.pool1(x)
            x = u1
            # if self.pool2 is not None:
            #     x = self.pool2(x)
            c3 = u2
            c4 = u3
            c5 = u4

            #down can be pretrained

            p5 = self.lateral_layer1(c5)
            p5_upsampled = self._upsample(p5, c4)
            p5 = self.corr_layer1(p5)
            p4 = self.lateral_layer2(c4)
            p4 = p5_upsampled + p4
            p4_upsampled = self._upsample(p4, c3)
            p4 = self.corr_layer2(p4)
            p3 = self.lateral_layer3(c3)
            p3 = p4_upsampled + p3
            p3 = self.corr_layer3(p3)
            p6 = self.conv6(c5)
            p7 = self.conv7(F.relu(p6))
            features = [p3, p4, p5, p6, p7]
            # ego_feat = self.avg_pool(p7)
            if self.pool2 is not None:
                for i in range(len(features)):
                    features[i] = self._upsample_time(features[i])
                # ego_feat = self._upsample_time(ego_feat)
        return features




    def identity_state_dict(self):
        state_dict = self.state_dict()
        logger.info('** Initilise identities **')
        for name, param in state_dict.items():
            if name.find('whh') > -1:
                logger.info('Set zeros for ' + name)
                p = param * 0.0
                if isinstance(param, Parameter):
                    # backwards compatibility for serialized parameters
                    p = param.data * 0.0
                # pdb.set_trace()
                for i in range(p.size(0)):
                    p[i, i, :, :, :] = 1.0/(p.size(2)*p.size(3)*p.size(4))
                try:
                    state_dict[name].copy_(p)
                    logger.info('Init' + name + ' ==>>' + str(p.size))
                except Exception:
                    raise RuntimeError('While copying the parameter named {}, '
                                       'whose dimensions in the model are {} and '
                                       'whose dimensions in the checkpoint are {}.'
                                       .format(name, own_state[name].size(), param.size()))


    def recurrent_conv_zero_state(self):
        state_dict = self.state_dict()
        print('**Initilise Recurrences for LSTMs or GRU **')
        for name, param in state_dict.items():
            if name.find('recurrent_conv') > -1:
                p = param * 0.0
                print('Set zeros for ' + name + str(torch.sum(torch.abs(param))))
                if isinstance(param, Parameter):
                    # backwards compatibility for serialized parameters
                    p = param.data * 0.0
                try:
                    state_dict[name].copy_(p)
                except Exception:
                    raise RuntimeError('While copying the parameter named {}, '
                                       'whose dimensions in the model are {} and '
                                       'whose dimensions in the checkpoint are {}.'
                                       .format(name, own_state[name].size(), param.size()))

                                       
    def load_my_state_dict(self, state_dict):
        logger.info('**LOAD STAE DICTIONARY FOR WEIGHT INITLISATIONS**')
        own_state = self.state_dict()
        update_name_dict = {}
        for b in range(len(self.non_local_inds)):
            count = -1
            for i in range(self.num_blocks[b]):
                count += 1
                sdlname = 'layer{:d}.{:d}'.format(b+1, i)
                oslname = 'layer{:d}.{:d}'.format(b+1, count)
                update_name_dict[oslname] = sdlname
                if i in self.non_local_inds[b]:
                    count += 1

        for own_name in own_state:
            state_name = own_name
            discard = False
            if own_name.startswith('layer'):
                sn = own_name.split('.')
                ln = sn[0]+'.'+sn[1]
                if ln in update_name_dict:
                    state_name = own_name.replace(ln, update_name_dict[ln])
                else:
                    discard = True
            # pdb.set_trace()
            if 'module.'+state_name in state_dict.keys():
                state_name = 'module.'+state_name 
            if state_name in state_dict.keys() and not discard:
                param = state_dict[state_name]
                own_size = own_state[own_name].size()
                param_size = param.size()
                match = True
                if len(param_size) != len(own_size) and len(own_size) > 1:
                    repeat_dim = own_size[2]
                    param = param.unsqueeze(2)
                    param = param.repeat(
                        1, 1, repeat_dim, 1, 1)/float(repeat_dim)
                else:
                    for i in range(len(param_size)):
                        if param_size[i] != own_size[i]:
                            match = False
                if match:
                    own_state[own_name].copy_(param)
                    logger.info(own_name + str(param_size) + ' ==>>' + str(own_size))
            else:
                logger.info('NAME IS NOT INPUT STATE DICT::>' + state_name)

def resnetfpn(args):
    model_type = args.MODEL_TYPE
    if model_type.startswith('I3D'):
        return ResNetFPN(BottleneckI3D, args)
    elif model_type.startswith('RCN'):
        return ResNetFPN(BottleneckRCN, args)
    elif model_type.startswith('2PD'):
        return ResNetFPN(Bottleneck2PD, args)
    elif model_type.startswith('C2D'):
        return ResNetFPN(BottleneckC2D, args)
    elif model_type.startswith('CLSTM'):
        return ResNetFPN(BottleneckCLSTM, args)
    elif model_type.startswith('RCLSTM'):
        return ResNetFPN(BottleneckRCLSTM, args)
    elif model_type.startswith('RCGRU'):
        return ResNetFPN(BottleneckRCGRU, args)
    elif model_type.startswith('SlowFast'):
        return ResNetFPN(BottleneckI3D, args)
    else:
        raise RuntimeError('Define the model type correctly:: ' + model_type)