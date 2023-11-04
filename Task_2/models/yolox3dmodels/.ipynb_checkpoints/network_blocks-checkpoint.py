#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn
from einops import rearrange

class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)



def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    elif name == 'swish':
        module = nn.SiLU(inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))




class BaseConv3D(nn.Module):
    """A Conv3d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, tksize=1,tstride=1,bias=False, act="silu",norm_type='bn'
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        tpad=(tksize-1)//2
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(tksize,ksize,ksize),
            stride=(tstride,stride,stride),
            padding=(tpad,pad,pad),
            groups=groups,
            bias=bias,
        )
        self.norm_type=norm_type  
        if norm_type == 'bn':
            self.bn = nn.BatchNorm3d(out_channels)
        elif norm_type == 'ln':
            self.ln=nn.LayerNorm(out_channels)
        if act == 'silu':
            self.act = get_activation(act, inplace=True)
        elif act =='GELU':
            self.act=nn.GELU()

    def forward(self, x):
        if self.norm_type == 'bn':
            return self.act(self.bn(self.conv(x).contiguous()))
        elif self.norm_type == 'ln':
            x=self.conv(x)
            x = rearrange(x, 'n c d h w -> n d h w c')
            x=self.ln(x)
            x = rearrange(x,'n d h w c -> n c d h w')
            return self.act(x)

    def fuseforward(self, x):
        return self.act(self.conv(x))


class BaseConv3DDINO(nn.Module):
    """A Conv3d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, tksize=1, tstride=1,bias=False, act="GELU",norm_type='ln'
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        tpad = (tksize - 1) // 2
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(tksize,ksize,ksize),
            stride=(tstride,stride,stride),
            padding=(tpad,pad,pad),
            groups=groups,
            bias=bias,
        )
        self.norm_type=norm_type  
        if norm_type == 'bn':
            self.bn = nn.BatchNorm3d(out_channels)
        elif norm_type == 'ln':
            self.ln=nn.LayerNorm(out_channels)
        if act == 'silu':
            self.act = get_activation(act, inplace=True)
        elif act =='GELU':
            self.act=nn.GELU()

    def forward(self, x):
        if self.norm_type == 'bn':
            return self.act(self.bn(self.conv(x)))
        elif self.norm_type == 'ln':
            x=self.conv(x)
            x = rearrange(x, 'n c d h w -> n d h w c')
            x=self.ln(x)
            x = rearrange(x,'n d h w c -> n c d h w')
            return self.act(x)

    def fuseforward(self, x):
        return self.act(self.conv(x))

class BaseTransConv3DDINO(nn.Module):
    """A Conv3d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1,tksize=1,tstride=1, bias=False, act="GELU",norm_type='ln'
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=(tksize,ksize,ksize),
            stride=(tstride,stride,stride),
            padding=(0,pad,pad),
            groups=groups,
            bias=bias,
        )
        self.norm_type=norm_type  
        if norm_type == 'bn':
            self.bn = nn.BatchNorm3d(out_channels)
        elif norm_type == 'ln':
            self.ln=nn.LayerNorm(out_channels)
        if act == 'silu':
            self.act = get_activation(act, inplace=True)
        elif act =='GELU':
            self.act=nn.GELU()

    def forward(self, x):
        if self.norm_type == 'bn':
            return self.act(self.bn(self.conv(x)))
        elif self.norm_type == 'ln':
            x=self.conv(x)
            x = rearrange(x, 'n c d h w -> n d h w c')
            x=self.ln(x)
            x = rearrange(x,'n d h w c -> n c d h w')
            return self.act(x)

    def fuseforward(self, x):
        return self.act(self.conv(x))

class BaseOnlyConv3D(nn.Module):
    """A Conv3d"""

    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, tksize=1, tstride=1,bias=False, act="silu"
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        tpad = (tksize - 1) // 2
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(tksize,ksize,ksize),
            stride=(tstride,stride,stride),
            padding=(tpad,pad,pad),
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        return self.conv(x)





class DWConv(nn.Module):
    """Depthwise Conv + Conv"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            act=act,
        )
        self.pconv = BaseConv(
            in_channels, out_channels, ksize=1, stride=1, groups=1, act=act
        )

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)




class DWConv3D(nn.Module):
    """Depthwise Conv + Conv"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv3D(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            act=act,
        )
        self.pconv = BaseConv3D(
            in_channels, out_channels, ksize=1, stride=1, groups=1, act=act
        )

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)









class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        expansion=0.5,
        tksize=1,tstride=1,
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1,tksize=tksize,tstride=tstride, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, tksize=tksize,tstride=tstride,act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y




class Bottleneck3D(nn.Module):
    # Standard bottleneck
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        super().__init__()
#         if tksize is None:
#             tksize = 1

#         if tstride is None:  
#             tstride = 1
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv3D if depthwise else BaseConv3D
        self.conv1 = BaseConv3D(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1,act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class Bottleneck3DDINO(nn.Module):
    # Standard bottleneck
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        expansion=0.5,
        tksize=3,
        tstride=1,
        depthwise=False,
        act="silu",
    ):
        super().__init__()
#         if tksize is None:
#             tksize = 1

#         if tstride is None:  
#             tstride = 1
        self.tksize=3
        self.tstride=1
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv3D if depthwise else BaseConv3D
        self.conv1 = BaseConv3D(in_channels, hidden_channels, 1, stride=1,tksize=self.tksize,tstride=self.tstride, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, tksize=self.tksize,tstride=self.tstride,act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y




class ResLayer(nn.Module):
    "Residual layer with `in_channels` inputs."

    def __init__(self, in_channels: int):
        super().__init__()
        mid_channels = in_channels // 2
        self.layer1 = BaseConv(
            in_channels, mid_channels, ksize=1, stride=1, act="lrelu"
        )
        self.layer2 = BaseConv(
            mid_channels, in_channels, ksize=3, stride=1, act="lrelu"
        )

    def forward(self, x):
        out = self.layer2(self.layer1(x))
        return x + out


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""

    def __init__(
        self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"
    ):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ]
        )
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class CSPLayer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(
        self,
        in_channels,
        out_channels,
        n=1,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [
            Bottleneck(
                hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act
            )
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)



class CSPLayer3D(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(
        self,
        in_channels,
        out_channels,
        n=1,
        shortcut=True,
        expansion=0.5,
        tksize=1,
        tstride=1,
        depthwise=False,
        act="silu",
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.tksize=tksize
        self.tstride=tstride
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv3D(in_channels, hidden_channels, 1, stride=1,tksize=tksize,tstride=tstride, act=act)
        self.conv2 = BaseConv3D(in_channels, hidden_channels, 1, stride=1, tksize=tksize,tstride=tstride,act=act)
        self.conv3 = BaseConv3D(2 * hidden_channels, out_channels, 1, stride=1,tksize=tksize,tstride=tstride, act=act)
        module_list = [
            Bottleneck3D(
                hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act
            )
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x).contiguous()
        x_1 = self.m(x_1).contiguous()
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)

class BaseOnlyTransConv3D(nn.Module):
    """A Conv3d"""

    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, tksize=1, tstride=1,bias=False, act="silu"
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        tpad = (tksize - 1) // 2
        self.conv = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=(tksize,ksize,ksize),
            stride=(tstride,stride,stride),
            padding=(tpad,pad,pad),
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        return self.conv(x)  
    
class CSPLayer3DDINO(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(
        self,
        in_channels,
        out_channels,
        n=1,
        shortcut=True,
        expansion=0.5,
        tksize=1,
        tstride=1,
        depthwise=False,
        act="silu",
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.tksize=tksize
        self.tstride=tstride
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv3D(in_channels, hidden_channels, 1, stride=1,tksize=tksize,tstride=tstride, act=act)
        self.conv2 = BaseConv3D(in_channels, hidden_channels, 1, stride=1, tksize=tksize,tstride=tstride,act=act)
        self.conv3 = BaseConv3D(2 * hidden_channels, out_channels, 1, stride=1,tksize=tksize,tstride=tstride, act=act)
        module_list = [
            Bottleneck3DDINO(
                hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act
            )
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)







class Focus(nn.Module):
    """Focus width and height information into channel space."""

    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return self.conv(x)
