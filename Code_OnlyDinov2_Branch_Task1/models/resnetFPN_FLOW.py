import torch.nn as nn
import math, pdb
import torch.nn.functional as F
import torch
from torch.nn.parameter import Parameter
from models.nonlocal_helper import Nonlocal
from models.blocks import *
from models.backbones import swin_transformer
import modules.utils as lutils
import pytorchvideo.layers as pvl
import yaml
from easydict import EasyDict
from .backbones import AVA_backbone

from functools import reduce, lru_cache
from operator import mul
from einops import rearrange
from .yolox3dmodels.yolo_pafpn import YOLOPAFPN3D

from .yolox3dmodels.darknet import CSPDarknet
from .yolox3dmodels.network_blocks import BaseConv3D, CSPLayer, DWConv,CSPLayer3D,BaseConv,DWConv3D,BaseOnlyConv3D

logger = lutils.get_logger(__name__)

### Download weights from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
def conv3x3(in_planes, out_planes, stride=1, padding=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1, 3, 3), stride=(1, stride, stride),
                     padding=(0,padding,padding), bias=bias)


def conv1x1(in_channel, out_channel):
    return nn.Conv3d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False)




class TSAttBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, ksize, stride, bias=False, 
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(1,ksize,ksize),
            stride=(1,stride,stride),
            padding=(0,pad,pad),
            bias=bias,
        )
        self.ln = nn.LayerNorm(out_channels)
        self.nonatt=pvl.create_nonlocal(dim_in=in_channels,dim_inner=in_channels//2)



    def forward(self, xrgb,xflow):
        x=torch.cat([xrgb,xflow],dim=1)
        x=self.conv(x)
        x=x.rearrange('n c d h w -> n d h w c')
        x=self.ln(x)
        x=x.rearrange('n d h w c -> n c d h w')
        return x




class ChannAT(nn.Module):
    """Constructs a Channel atte module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self,in_channels,b=1,gamma=2,init_zero=True):
        super(ChannAT, self).__init__()
        kernel_size=int(abs((math.log(in_channels,2)+b)/gamma))
        kernel_size=kernel_size if kernel_size%2 else kernel_size+1
        # print("kernel_size",kernel_size)
        padding=kernel_size//2
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conveca = nn.Conv2d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)
        if init_zero:
            nn.init.zeros_(self.conveca.weight)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, t, h, w = x.size()
        y = self.avg_pool(x).view([b,1,c,1])
        y = self.conveca(y).view([b,c,1,1,1])

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y + x




class TSAttBlock(nn.Module):
    def __init__(
        self, rgb_in_channels,flow_in_channels ,out_channels, ksize, stride, bias=False,zero_init_final_norm=True,zero_init_final_conv=True,non_local=False,uptemporal=True 
    ):
        super(TSAttBlock,self).__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv3d(
            (rgb_in_channels+flow_in_channels),
            out_channels,
            kernel_size=(1,ksize,ksize),
            stride=(1,stride,stride),
            padding=(0,pad,pad),
            bias=bias,
        )
        self.eca=ChannAT(in_channels=(rgb_in_channels+flow_in_channels))
        if non_local:
            self.nonatt=pvl.create_nonlocal(dim_in=out_channels,dim_inner=out_channels//2,zero_init_final_norm=zero_init_final_norm,zero_init_final_conv=zero_init_final_conv)
        self.non_local=non_local
        self.uptemporal=uptemporal
        if uptemporal:
            self.upconv=nn.ConvTranspose3d(
                out_channels,
                out_channels,
                kernel_size=(2,ksize,ksize),
                stride=(2,stride,stride),
                padding=(0,pad,pad),
                bias=bias,
            )


    def forward(self, xrgb,xflow):
        x=torch.cat([xrgb,xflow],dim=1)
        x=self.eca(x)
        x=self.conv(x)
        if self.non_local:
            x=self.nonatt(x)
        if self.uptemporal:
            x=self.upconv(x)
        return (x)



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
        
        self.swinaclayer1=swin_transformer.BasicLayer(dim=int(embed_dim*2**0),depth=depths[0],num_heads=num_heads[0],window_size=self.window_size,
                                                      qkv_bias=True,drop_path=dpr[sum(depths[:0]):sum(depths[:1])],downsample=swin_transformer.PatchMerging if 0<self.num_layers-1 else None)#384 4 28 28
        
        self.swinaclayer2=swin_transformer.BasicLayer(dim=embed_dim*2**1,depth=2,num_heads=12,window_size=(8,7,7),
                                                      qkv_bias=True,drop_path=dpr[sum(depths[:1]):sum(depths[:2])],downsample=swin_transformer.PatchMerging if 1<self.num_layers-1 else None)

        self.swinaclayer3=swin_transformer.BasicLayer(dim=embed_dim*2**2,depth=18,num_heads=24,window_size=(8,7,7),
                                                      qkv_bias=True,drop_path=dpr[sum(depths[:2]):sum(depths[:3])],downsample=swin_transformer.PatchMerging if 2<self.num_layers-1 else None)#768 4 14 14
    
        self.swinaclayer4=swin_transformer.BasicLayer(dim=embed_dim*2**3,depth=2,num_heads=48,window_size=(8,7,7),
                                                      qkv_bias=True,drop_path=dpr[sum(depths[:3]):sum(depths[:3 + 1])],downsample=swin_transformer.PatchMerging if 3<self.num_layers-1 else None)#1536 4 7 7

        self.norm3 = nn.LayerNorm(self.num_features)

        self._freeze_stages()
        self.layer_names = []


        self.layer1 = self._make_layer(
            block, 64, num_blocks[0], temp_kernals=model_3d_layers[0], nl_inds=non_local_inds[0])
        self.MODEL_TYPE = args.MODEL_TYPE
        self.pool2=None
        self.layer2 = self._make_layer(
            block, 128, num_blocks[1], stride=2, temp_kernals=model_3d_layers[1], nl_inds=non_local_inds[1])
        self.layer3 = self._make_layer(
            block, 256, num_blocks[2], stride=2, temp_kernals=model_3d_layers[2], nl_inds=non_local_inds[2])
        self.layer4 = self._make_layer(
            block, 512, num_blocks[3], stride=2, temp_kernals=model_3d_layers[3], nl_inds=non_local_inds[3])

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


    
class TSVideoSwinBackbone(nn.Module):

    def __init__(self,   pretrained=None,
                 pretrained2d=True,
                 patch_size=(2, 4, 4),
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=(16,7,7),
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
        
        super(TSVideoSwinBackbone, self).__init__()
    
    
        self.rgbbackbone=viedoswinRGB()
        self.flowbackbone=viedoswinFlow()
        self.catconv1=TSAttBlock(rgb_in_channels=256,flow_in_channels=256,out_channels=256,ksize=1,stride=1)
        self.catconv2=TSAttBlock(rgb_in_channels=512,flow_in_channels=512,out_channels=512,ksize=1,stride=1)
        self.catconv3=TSAttBlock(rgb_in_channels=1024,flow_in_channels=1024,out_channels=1024,ksize=1,stride=1)
        

    def forward(self, xrgb,xflow):
            
        xrgb2,xrgb3,xrgb4=self.rgbbackbone(xrgb)
        xflow2,xflow3,xflow4=self.flowbackbone(xflow)
        x2=self.catconv1(xrgb2,xflow2)
        x3=self.catconv2(xrgb3,xflow3)
        x4=self.catconv3(xrgb4,xflow4)
        # print("catconv1",x2.shape)
        # print("catconv2",x3.shape)
        # print("catconv3",x4.shape)
        return [x2,x3,x4]


class YOLOPAFPN3D(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("dark3", "dark4", "dark5"),
        in_channels=[256, 512, 1024],
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        self.backbone = TSVideoSwinBackbone()
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv3D if depthwise else BaseConv3D
        width=1.0
        depth=1.0
        self.upsample = nn.Upsample(scale_factor=(1,2,2), mode="nearest")
        self.lateral_conv0 = BaseConv3D(
            int(in_channels[2] * width), int(in_channels[1] * depth), 1, 1, act=act
        )
        self.C3_p4 = CSPLayer3D(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv3D(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer3D(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer3D(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer3D(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )
        self.upcovhead1=BaseOnlyConv3D(
            512,256, 1, 1, act=act
        )
        self.upcovhead2=BaseOnlyConv3D(
            1024,256, 1, 1, act=act
        )
        self.upcovhead3=BaseOnlyConv3D(
            1024,256, ksize=3, stride=2, act=act
        )
        self.upcovhead4=BaseOnlyConv3D(
            256,256, ksize=3, stride=2, act=act
        )

    def forward(self, RGBimages,Flowimages):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        #  backbone
        out_features = self.backbone(RGBimages,Flowimages)
    
        [x2, x1, x0] = out_features

        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32
        #p7 5 5 256,p6 10 10 256 , p5 20 20 256, p4 40 40 256, p3 80 80 256
        #pan0 20 20 1024, pan1 40 40 512, pan2 80 80 256
        p3=pan_out2
        p4=self.upcovhead1(pan_out1)
        p5=self.upcovhead2(pan_out0)
        p6=self.upcovhead3(pan_out0)
        p7=self.upcovhead4(p6)
        outputs = [p3, p4, p5, p6, p7]
        return outputs
    

class viedoswinFlow(nn.Module):

    def __init__(self,  pretrained=None,
                 pretrained2d=True,
                 patch_size=(2, 4, 4),
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=(16,7,7),
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
                 drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.4,
        patch_norm=True,
                 norm_layer=nn.LayerNorm,
                 frozen_stages=-1,
                 use_checkpoint=False):
        
        self.inplanes = 64

        super(viedoswinFlow, self).__init__()
    


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule


        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.frozen_stages = frozen_stages
        self.window_size = window_size
        self.patch_size = patch_size

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
        
        
        self.swinaclayer2=swin_transformer.BasicLayer(dim=embed_dim*2**1,depth=depths[1],num_heads=num_heads[1],window_size=self.window_size,
                                                      qkv_bias=True,drop_path=dpr[sum(depths[:1]):sum(depths[:2])],downsample=swin_transformer.PatchMerging if 1<self.num_layers-1 else None)
        
        
        self.swinaclayer3=swin_transformer.BasicLayer(dim=embed_dim*2**2,depth=depths[2],num_heads=num_heads[2],window_size=self.window_size,
                                                      qkv_bias=True,drop_path=dpr[sum(depths[:2]):sum(depths[:3])],downsample=swin_transformer.PatchMerging if 2<self.num_layers-1 else None)#768 4 14 14
        
        
        self.swinaclayer4=swin_transformer.BasicLayer(dim=embed_dim*2**3,depth=depths[3],num_heads=num_heads[3],window_size=self.window_size,
                                                      qkv_bias=True,drop_path=dpr[sum(depths[:3]):sum(depths[:3 + 1])],downsample=swin_transformer.PatchMerging if 3<self.num_layers-1 else None)#1536 4 7 7

        self.norm3 = nn.LayerNorm(self.num_features)

        self._freeze_stages()
        

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


    def forward(self, x):
        x=self.patch_embed(x)
        x=self.pos_drop(x)

        x,xbfd1=self.swinaclayer1(x)
        x,xbfd2=self.swinaclayer2(x)
        x,xbfd3=self.swinaclayer3(x)
        x,xbfd4=self.swinaclayer4(x)

        x = rearrange(x, 'n c d h w -> n d h w c')
        x = self.norm3(x)
        x = rearrange(x, 'n d h w c -> n c d h w')

        return [xbfd2,xbfd3,x]


class viedoswinRGB(nn.Module):

    def __init__(self,  pretrained=None,
                 pretrained2d=True,
                 patch_size=(2, 4, 4),
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=(16,7,7),
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
        super(viedoswinRGB, self).__init__()
    
    


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule


        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.frozen_stages = frozen_stages
        self.window_size = window_size
        self.patch_size = patch_size




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
        
        
        self.swinaclayer2=swin_transformer.BasicLayer(dim=embed_dim*2**1,depth=depths[1],num_heads=num_heads[1],window_size=self.window_size,
                                                      qkv_bias=True,drop_path=dpr[sum(depths[:1]):sum(depths[:2])],downsample=swin_transformer.PatchMerging if 1<self.num_layers-1 else None)
        

        
        self.swinaclayer3=swin_transformer.BasicLayer(dim=embed_dim*2**2,depth=depths[2],num_heads=num_heads[2],window_size=self.window_size,
                                                      qkv_bias=True,drop_path=dpr[sum(depths[:2]):sum(depths[:3])],downsample=swin_transformer.PatchMerging if 2<self.num_layers-1 else None)#768 4 14 14
        

        
        self.swinaclayer4=swin_transformer.BasicLayer(dim=embed_dim*2**3,depth=depths[3],num_heads=num_heads[3],window_size=self.window_size,
                                                      qkv_bias=True,drop_path=dpr[sum(depths[:3]):sum(depths[:3 + 1])],downsample=swin_transformer.PatchMerging if 3<self.num_layers-1 else None)#1536 4 7 7
        # self.layers.append(layer)

        self.norm3 = nn.LayerNorm(self.num_features)

        self._freeze_stages()
        

    def _freeze_stages(self):
        # if self.frozen_stages >= 0:
        #     self.patch_embed.eval()
        #     for param in self.patch_embed.parameters():
        #         param.requires_grad = False

        if self.frozen_stages >= 1:
            self.pos_drop.eval()
            for i in range(1, self.frozen_stages + 1):
                layer_name = 'swinaclayer{}'.format(i)
                layert = getattr(self, layer_name)
                m = layert
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False


    def forward(self, x):
        x=self.patch_embed(x)
        x=self.pos_drop(x)

        x,xbfd1=self.swinaclayer1(x)
        x,xbfd2=self.swinaclayer2(x)
        x,xbfd3=self.swinaclayer3(x)
        x,xbfd4=self.swinaclayer4(x)

        x = rearrange(x, 'n c d h w -> n d h w c')
        x = self.norm3(x)
        x = rearrange(x, 'n d h w c -> n c d h w')

        return [xbfd2,xbfd3,x]


                                       
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