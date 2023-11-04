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
from .yolox3dmodels.network_blocks import BaseConv3D, CSPLayer, DWConv,CSPLayer3D,BaseTransConv3DDINO,DWConv3D,BaseOnlyConv3D,BaseConv3DDINO,CSPLayer3DDINO,BaseOnlyTransConv3D
from mmpretrain import get_model
from mmaction.apis import inference_recognizer, init_recognizer
logger = lutils.get_logger(__name__)

### Download weights from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
def conv3x3(in_planes, out_planes, stride=1, padding=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1, 3, 3), stride=(1, stride, stride),
                     padding=(0,padding,padding), bias=bias)


def conv1x1(in_channel, out_channel):
    return nn.Conv3d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False)



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
        print("kernel_size",kernel_size)
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
        self, rgb_in_channels,flow_in_channels ,out_channels, ksize, stride, bias=False,tksize=1,tstride=1,zero_init_final_norm=True,zero_init_final_conv=True,non_local=False,uptemporal=False 
    ):
        super(TSAttBlock,self).__init__()
        # same padding
        pad = (ksize - 1) // 2
        tpad = (tksize - 1) // 2
        self.conv = nn.Conv3d(
            (rgb_in_channels+flow_in_channels),
            out_channels,
            kernel_size=(tksize,ksize,ksize),
            stride=(tstride,stride,stride),
            padding=(tpad,pad,pad),
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


class TimeGlobalMultiScaleBlock(nn.Module):
    def __init__(self,in_channels,out_channels,trans=False):
        super(TimeGlobalMultiScaleBlock,self).__init__()
        self.tconv1=BaseOnlyConv3D(in_channels,out_channels,1,1,tksize=3,tstride=1,bias=True)
        self.tconv2=BaseOnlyConv3D(in_channels,out_channels,1,1,tksize=5,tstride=1,bias=True)
        self.tconv3=BaseOnlyConv3D(in_channels,out_channels,1,1,tksize=7,tstride=1,bias=True)
        self.trans=trans
        if trans==False:
            self.tavg=BaseOnlyConv3D(out_channels*3,out_channels,1,1,tksize=1,tstride=1,bias=False)
            nn.init.zeros_(self.tavg.conv.weight)
        else:
            self.tavg=BaseOnlyTransConv3D(out_channels*3,out_channels,1,1,tksize=2,tstride=2,bias=False)
            self.upconv=BaseOnlyTransConv3D(out_channels,out_channels,1,1,tksize=2,tstride=2,bias=False)
            nn.init.zeros_(self.tavg.conv.weight)
        self.norm=nn.LayerNorm(out_channels)
        self.act=nn.GELU()

    def forward(self,x):
        xinit=x
        x1=self.tconv1(x)
        x2=self.tconv2(x)
        x3=self.tconv3(x)
        xnew=torch.cat([x1,x2,x3],dim=1)
        xnew=self.tavg(xnew)
        x=self.act(xnew)
        x = rearrange(x, 'n c d h w -> n d h w c')
        x = self.norm(x)
        x = rearrange(x, 'n d h w c -> n c d h w')
        if self.trans==False:
            return x+xinit
        else:
            return x+self.upconv(xinit)
    

    
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
    
    
        self.rgbbackbone=viedoDINORGB()
        self.tconv=TimeGlobalMultiScaleBlock(1024,1024)
        self.tconv_1=TimeGlobalMultiScaleBlock(256,256)
        self.tconv_2=TimeGlobalMultiScaleBlock(512,512)
        self.tconv_3=TimeGlobalMultiScaleBlock(1024,1024)


        self.conv2=BaseConv3DDINO(1024,512,3,1) # for 512 40 40
        self.deconv3=BaseTransConv3DDINO(1024,256,3,1)# for 256 80 80
        # self.norm1=nn.LayerNorm(1024)
        # self.norm2=nn.LayerNorm(512)
        # self.norm3=nn.LayerNorm(256)
        self.flowbackbone=viedoswinFlow()
        self.uptime1=nn.ConvTranspose3d(384,256,kernel_size=(2,1,1), stride=(2,1,1))
        self.uptime2=nn.ConvTranspose3d(768,512,kernel_size=(2,1,1), stride=(2,1,1))
        self.uptime3=nn.ConvTranspose3d(1536,1024,kernel_size=(2,1,1), stride=(2,1,1))
        self.catconv1=TSAttBlock(rgb_in_channels=256,flow_in_channels=256,out_channels=256,ksize=1,stride=1)
        self.catconv2=TSAttBlock(rgb_in_channels=512,flow_in_channels=512,out_channels=512,ksize=1,stride=1)
        self.catconv3=TSAttBlock(rgb_in_channels=1024,flow_in_channels=1024,out_channels=1024,ksize=1,stride=1)
        
        

    def forward(self, xrgb):
        xsw1,xsw2,xsw3=self.flowbackbone(xrgb)#1 t/2 1152 16 22
        xsw1=self.uptime1(xsw1)#384
        xsw2=self.uptime2(xsw2)#768
        xsw3=self.uptime3(xsw3)#1536
        cls_token,out_feat=self.rgbbackbone(xrgb)
        out_feat=self.tconv(out_feat)
        out_feat_1=out_feat
        out_feat_1=F.interpolate(out_feat_1, size=(out_feat_1.shape[2],16,22), mode='trilinear')#1536
        out_feat_2=self.conv2(out_feat)
        out_feat_2=F.interpolate(out_feat_2, size=(out_feat_2.shape[2],32,44), mode='trilinear')#768
        out_feat_3=self.deconv3(out_feat)
        out_feat_3=F.interpolate(out_feat_3, size=(out_feat_3.shape[2],64,88), mode='trilinear')#384
 
        x2=self.catconv1(out_feat_3,xsw1)#256
        x3=self.catconv2(out_feat_2,xsw2)#512
        x4=self.catconv3(out_feat_1,xsw3)#1024
        x2=self.tconv_1(x2)
        x3=self.tconv_2(x3)
        x4=self.tconv_3(x4)

        return [cls_token,x2,x3,x4]


class YOLOPAFPN3DDINO(nn.Module):
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
        super(YOLOPAFPN3DDINO,self).__init__()
        self.backbone = TSVideoSwinBackbone()
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv3D if depthwise else BaseConv3D
        width=1.0
        depth=1.0
        self.upsample = nn.Upsample(scale_factor=(1,2,2), mode="nearest")
        self.lateral_conv0 = BaseConv3D(
            int(in_channels[2] * width), int(in_channels[1] * depth), 1, 1,tksize=3,tstride=1, act=act
        )
        self.C3_p4 = CSPLayer3DDINO(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            tksize=3,tstride=1,
            act=act,
        )  # cat  and time

        self.reduce_conv1 = BaseConv3D(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, tksize=3,tstride=1,act=act
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
        self.C3_n4 = CSPLayer3DDINO(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            tksize=3,tstride=1,
            act=act,
        )# time conv
        self.upcovhead1=BaseOnlyConv3D(
            512,256, 1, 1,tksize=3, tstride=1, act=act
        )
        self.upcovhead2=BaseOnlyConv3D(
            1024,256, 1, 1, tksize=3, tstride=1,act=act
        )
        self.upcovhead3=BaseOnlyConv3D(
            1024,256, ksize=3, stride=2,tksize=3, tstride=1, act=act
        )
        self.upcovhead4=BaseOnlyConv3D(
            256,256, ksize=3, stride=2,tksize=3, tstride=1, act=act
        )

    def forward(self, RGBimages):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        #  backbone
        out_features = self.backbone(RGBimages)
    
        [cls_token,x2, x1, x0] = out_features

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
        return cls_token,outputs
    



class viedoswinFlow(nn.Module):

    def __init__(self,  pretrained=None,
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
        
        # self.patchmerge1=swin_transformer.PatchMerging(dim=192)#384 4 28 28
        
        self.swinaclayer2=swin_transformer.BasicLayer(dim=embed_dim*2**1,depth=depths[1],num_heads=num_heads[1],window_size=self.window_size,
                                                      qkv_bias=True,drop_path=dpr[sum(depths[:1]):sum(depths[:2])],downsample=swin_transformer.PatchMerging if 1<self.num_layers-1 else None)
        
        # self.patchmerge2=swin_transformer.PatchMerging(dim=192*2)#768 4 14 14
        
        self.swinaclayer3=swin_transformer.BasicLayer(dim=embed_dim*2**2,depth=depths[2],num_heads=num_heads[2],window_size=self.window_size,
                                                      qkv_bias=True,drop_path=dpr[sum(depths[:2]):sum(depths[:3])],downsample=swin_transformer.PatchMerging if 2<self.num_layers-1 else None)#768 4 14 14
        
        # self.patchmerge3=swin_transformer.PatchMerging(dim=192*2**2)#1536 4 7 7
        
        self.swinaclayer4=swin_transformer.BasicLayer(dim=embed_dim*2**3,depth=depths[3],num_heads=num_heads[3],window_size=self.window_size,
                                                      qkv_bias=True,drop_path=dpr[sum(depths[:3]):sum(depths[:3 + 1])],downsample=swin_transformer.PatchMerging if 3<self.num_layers-1 else None)#1536 4 7 7
        # self.layers.append(layer)

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

        x,xbfd2=self.swinaclayer2(x)#xbfd2 384

        x,xbfd3=self.swinaclayer3(x)#xbfd3 768
        x,xbfd4=self.swinaclayer4(x)# x 1535
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
    


modeldinobackbone=get_model('vit-large-p14_dinov2-pre_3rdparty',device='cpu', pretrained=True,backbone=dict(out_type='cls_token_and_featmap')).backbone
class viedoDINORGB(nn.Module):

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
        
        super(viedoDINORGB, self).__init__()
        self.dinobackbone=modeldinobackbone
        self.dinobackbone.eval()
        for param in self.dinobackbone.parameters():
            param.requires_grad = False
    


    def forward(self, x):
        x_splits=torch.chunk(x, x.size(2), dim=2) 
        out_splits_cls = []
        out_splits_feat = []
        for split in x_splits:
            split=split[:,:,0,:,:]
            out = self.dinobackbone(split)
            out_splits_cls.append(out[0][0])
            out_splits_feat.append(out[0][1])

        # 拼接回原始形状
        out_cls_token = torch.stack(out_splits_cls, dim=1) # end cls token dino
        out_feat = torch.stack(out_splits_feat, dim=2)# end feat dino 1024 t 46 46 if 640*640

        return [out_cls_token,out_feat]





                                       
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