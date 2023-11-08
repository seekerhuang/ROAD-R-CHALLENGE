

""" 

FPN network Classes

Author: Gurkirt Singh
Inspired from https://github.com/kuangliu/pytorch-retinanet and
https://github.com/gurkirt/realtime-action-detection

"""

import math

import torch
import torch.nn as nn

import modules.utils as utils
from models.backbone_models import backbone_models
from modules.anchor_box_kmeans import anchorBox as KanchorBox
from modules.anchor_box_retinanet import anchorBox as RanchorBox
from modules.box_utils import decode
from modules.detection_loss import FocalLoss
from models.resnetFPN_FLOW import YOLOPAFPN3D
from models.resnetFPN_DINOV2 import YOLOPAFPN3DDINO
# from torch.cuda.amp import autocast as autocast

logger = utils.get_logger(__name__)



class dinov2linear(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linearcls1 = nn.Linear(dim, dim)
        self.liclsact1 = nn.GELU()
        nn.init.zeros_(self.linearcls1.weight)
        nn.init.zeros_(self.linearcls1.bias)
        self.linearcls2 = nn.Linear(dim, dim)
        nn.init.zeros_(self.linearcls2.weight)
        nn.init.zeros_(self.linearcls2.bias)
        self.liclsact2 = nn.GELU()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
    def forward(self, x):
        xinit=x
        x = self.linearcls1(x)
        x=self.norm1(x)
        x = self.liclsact1(x)
        x = self.linearcls2(x)
        x=self.norm2(x)
        x = self.liclsact2(x)
        return (xinit + x)

    
class dinov2linearFix(nn.Module):
    def __init__(self, dimin,dimout):
        super().__init__()
        self.linearcls1 = nn.Linear(dimin, dimout)
        self.liclsact1 = nn.GELU()
        self.linearcls2 = nn.Linear(dimout, dimout)
        nn.init.zeros_(self.linearcls2.weight)
        nn.init.zeros_(self.linearcls2.bias)
        self.norm1 = nn.LayerNorm(dimout)
        self.norm2 = nn.LayerNorm(dimout)
        self.liclsact2 = nn.GELU()
        
    def forward(self, x):
        x = self.linearcls1(x)
        x=self.norm1(x)
        x = self.liclsact1(x)
        xmid=x
        x = self.linearcls2(x)
        x=self.norm2(x)
        x = self.liclsact2(x)
        return (x+xmid)




class RetinaNet(nn.Module):
    """Feature Pyramid Network Architecture
    The network is composed of a backbone FPN network followed by the
    added Head conv layers.  
    Each head layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
    See: 
    RetinaNet: https://arxiv.org/pdf/1708.02002.pdf for more details.
    FPN: https://arxiv.org/pdf/1612.03144.pdf

    Args:
        backbone Network:
        Program Argument Namespace

    """

    def __init__(self, backbone, args):
        super(RetinaNet, self).__init__()

        self.num_classes = args.num_classes
        # TODO: implement __call__ in

        if args.ANCHOR_TYPE == 'RETINA':
            self.anchors = RanchorBox()
        elif args.ANCHOR_TYPE == 'KMEANS':
            self.anchors = KanchorBox()
        else:
            raise RuntimeError('Define correct anchor type')
            
        # print('Cell anchors\n', self.anchors.cell_anchors)
        # pdb.set_trace()
        self.ar = self.anchors.ar
        args.ar = self.ar
        self.use_bias = True
        self.head_size = args.head_size
        self.backbone = backbone
        self.SEQ_LEN = args.SEQ_LEN
        self.HEAD_LAYERS = args.HEAD_LAYERS
        self.NUM_FEATURE_MAPS = args.NUM_FEATURE_MAPS
        
        self.reg_heads = []
        self.cls_heads = []
        self.prior_prob = 0.01
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        # for nf in range(self.NUM_FEATURE_MAPS):
        self.reg_heads = self.make_head(
            self.ar * 4, args.REG_HEAD_TIME_SIZE, self.HEAD_LAYERS)
        self.cls_heads = self.make_head(
            self.ar * self.num_classes, args.CLS_HEAD_TIME_SIZE, self.HEAD_LAYERS)
        
        nn.init.constant_(self.cls_heads[-1].bias, bias_value)

        if args.MODE == 'train':  # eval_iters only in test case
            self.criterion = FocalLoss(args)



    # def forward(self, images, gt_boxes=None, gt_labels=None, ego_labels=None, counts=None, img_indexs=None, get_features=False):
    def forward(self, images, gt_boxes=None, gt_labels=None, counts=None, img_indexs=None, get_features=False, logic=None, Cplus=None, Cminus=None, is_ulb=None):
        # sources, ego_feat = self.backbone(images)
        sources = self.backbone(images)

        grid_sizes = [feature_map.shape[-2:] for feature_map in sources]
        ancohor_boxes = self.anchors(grid_sizes)
        
        loc = list()
        conf = list()

        for x in sources:
            loc.append(self.reg_heads(x).permute(0, 2, 3, 4, 1).contiguous())
            conf.append(self.cls_heads(x).permute(0, 2, 3, 4, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), o.size(1), -1) for o in loc], 2)
        conf = torch.cat([o.view(o.size(0), o.size(1), -1) for o in conf], 2)

        flat_loc = loc.view(loc.size(0), loc.size(1), -1, 4)
        flat_conf = conf.view(conf.size(0), conf.size(1), -1, self.num_classes)

        # pdb.set_trace()
        if get_features:  # testing mode with feature return
            return flat_conf, features
        elif gt_boxes is not None:  # training mode
          
            return self.criterion(flat_conf, flat_loc, gt_boxes, gt_labels, counts, ancohor_boxes, logic, Cplus, Cminus, is_ulb=is_ulb), is_ulb

        else:  # otherwise testing mode
            decoded_boxes = []
            for b in range(flat_loc.shape[0]):
                temp_l = []
                for s in range(flat_loc.shape[1]):
                    # torch.stack([decode(flat_loc[b], ancohor_boxes) for b in range(flat_loc.shape[0])], 0),
                    temp_l.append(decode(flat_loc[b, s], ancohor_boxes))
                decoded_boxes.append(torch.stack(temp_l, 0))
            # return torch.stack(decoded_boxes, 0), flat_conf, ego_preds
            return torch.stack(decoded_boxes, 0), flat_conf


    def make_features(self,  shared_heads):
        layers = []
        use_bias = self.use_bias
        head_size = self.head_size
        for _ in range(shared_heads):
            layers.append(nn.Conv3d(head_size, head_size, kernel_size=(
                1, 3, 3), stride=1, padding=(0, 1, 1), bias=use_bias))
            layers.append(nn.ReLU(True))

        layers = nn.Sequential(*layers)

        for m in layers.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if hasattr(m.bias, 'data'):
                    nn.init.constant_(m.bias, 0)

        return layers

    def make_head(self, out_planes, time_kernel,  num_heads_layers):
        layers = []
        use_bias = self.use_bias
        head_size = self.head_size
        
        for kk in range(num_heads_layers):
            branch_kernel = 1
            bpad = 0
            layers.append(nn.Conv3d(head_size, head_size, kernel_size=(
                branch_kernel, 3, 3), stride=1, padding=(bpad, 1, 1), bias=use_bias))
            layers.append(nn.GELU())

        tpad = time_kernel//2
        layers.append(nn.Conv3d(head_size, out_planes, kernel_size=(
            time_kernel, 3, 3), stride=1, padding=(tpad, 1, 1)))
        
        layers = nn.Sequential(*layers)

        for m in layers.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if hasattr(m.bias, 'data'):
                    nn.init.constant_(m.bias, 0)

        return layers

class SwinRetinaNet(nn.Module):
    """Feature Pyramid Network Architecture
    The network is composed of a backbone FPN network followed by the
    added Head conv layers.  
    Each head layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
    See: 
    RetinaNet: https://arxiv.org/pdf/1708.02002.pdf for more details.
    FPN: https://arxiv.org/pdf/1612.03144.pdf

    Args:
        backbone Network:
        Program Argument Namespace

    """

    def __init__(self, backbone, args):
        super(SwinRetinaNet, self).__init__()

        self.num_classes = args.num_classes
        # TODO: implement __call__ in

        if args.ANCHOR_TYPE == 'RETINA':
            self.anchors = RanchorBox()
        elif args.ANCHOR_TYPE == 'KMEANS':
            self.anchors = KanchorBox()
        else:
            raise RuntimeError('Define correct anchor type')
            
        # print('Cell anchors\n', self.anchors.cell_anchors)
        # pdb.set_trace()
        self.ar = self.anchors.ar
        args.ar = self.ar
        self.use_bias = True
        self.head_size = args.head_size
        self.backbone = YOLOPAFPN3D()
        self.SEQ_LEN = args.SEQ_LEN
        self.HEAD_LAYERS = args.HEAD_LAYERS
        self.NUM_FEATURE_MAPS = args.NUM_FEATURE_MAPS
        
        self.reg_heads = []
        self.cls_heads = []
        self.prior_prob = 0.01
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        # for nf in range(self.NUM_FEATURE_MAPS):
        self.reg_heads = self.make_head(
            self.ar * 4, args.REG_HEAD_TIME_SIZE, self.HEAD_LAYERS)
        self.cls_heads = self.make_head(
            self.ar * self.num_classes, args.CLS_HEAD_TIME_SIZE, self.HEAD_LAYERS)
        
        nn.init.constant_(self.cls_heads[-1].bias, bias_value)

        if args.MODE == 'train':  # eval_iters only in test case
            self.criterion = FocalLoss(args)


    # def forward(self, images, gt_boxes=None, gt_labels=None, ego_labels=None, counts=None, img_indexs=None, get_features=False):
    def forward(self, RGBimages,Flowimages ,gt_boxes=None, gt_labels=None, counts=None, img_indexs=None, get_features=False, logic=None, Cplus=None, Cminus=None, is_ulb=None):
        # sources, ego_feat = self.backbone(images)
        sources = self.backbone(RGBimages,Flowimages)


        grid_sizes = [feature_map.shape[-2:] for feature_map in sources]
        ancohor_boxes = self.anchors(grid_sizes)
        
        loc = list()
        conf = list()

        for x in sources:
            loc.append(self.reg_heads(x).permute(0, 2, 3, 4, 1).contiguous())
            conf.append(self.cls_heads(x).permute(0, 2, 3, 4, 1).contiguous())
        # loc b t h w c
        # conf b t h w c
        loc = torch.cat([o.view(o.size(0), o.size(1), -1) for o in loc], 2)
        #loc b t hwc+ hwc+ hwc+ hwc+ hwc
        conf = torch.cat([o.view(o.size(0), o.size(1), -1) for o in conf], 2)

        flat_loc = loc.view(loc.size(0), loc.size(1), -1, 4)
        flat_conf = conf.view(conf.size(0), conf.size(1), -1, self.num_classes)

        # pdb.set_trace()
        if get_features:  # testing mode with feature return
            return flat_conf, features
        elif gt_boxes is not None:  # training mode
           
            return self.criterion(flat_conf, flat_loc, gt_boxes, gt_labels, counts, ancohor_boxes, logic, Cplus, Cminus, is_ulb=is_ulb), is_ulb

        else:  # otherwise testing mode
            decoded_boxes = []
            for b in range(flat_loc.shape[0]):
                temp_l = []
                for s in range(flat_loc.shape[1]):
                    # torch.stack([decode(flat_loc[b], ancohor_boxes) for b in range(flat_loc.shape[0])], 0),
                    temp_l.append(decode(flat_loc[b, s], ancohor_boxes))
                decoded_boxes.append(torch.stack(temp_l, 0))
            # return torch.stack(decoded_boxes, 0), flat_conf, ego_preds
            return torch.stack(decoded_boxes, 0), flat_conf


    def make_features(self,  shared_heads):
        layers = []
        use_bias = self.use_bias
        head_size = self.head_size
        for _ in range(shared_heads):
            layers.append(nn.Conv3d(head_size, head_size, kernel_size=(
                1, 3, 3), stride=1, padding=(0, 1, 1), bias=use_bias))
            layers.append(nn.ReLU(True))

        layers = nn.Sequential(*layers)

        for m in layers.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if hasattr(m.bias, 'data'):
                    nn.init.constant_(m.bias, 0)

        return layers

    def make_head(self, out_planes, time_kernel,  num_heads_layers):
        layers = []
        use_bias = self.use_bias
        head_size = self.head_size
        
        for kk in range(num_heads_layers):
      
            branch_kernel = 1
            bpad = 0
            layers.append(nn.Conv3d(head_size, head_size, kernel_size=(
                branch_kernel, 3, 3), stride=1, padding=(bpad, 1, 1), bias=use_bias))
            layers.append(nn.ReLU(True))

        tpad = time_kernel//2
        layers.append(nn.Conv3d(head_size, out_planes, kernel_size=(
            time_kernel, 3, 3), stride=1, padding=(tpad, 1, 1)))
        
        layers = nn.Sequential(*layers)

        for m in layers.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if hasattr(m.bias, 'data'):
                    nn.init.constant_(m.bias, 0)

        return layers
    
    
    
class DinoRetinaNet(nn.Module):
    """Feature Pyramid Network Architecture
    The network is composed of a backbone FPN network followed by the
    added Head conv layers.  
    Each head layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
    See: 
    RetinaNet: https://arxiv.org/pdf/1708.02002.pdf for more details.
    FPN: https://arxiv.org/pdf/1612.03144.pdf

    Args:
        backbone Network:
        Program Argument Namespace

    """

    def __init__(self, backbone, args):
        super(DinoRetinaNet, self).__init__()

        self.num_classes = args.num_classes
        # TODO: implement __call__ in

        if args.ANCHOR_TYPE == 'RETINA':
            self.anchors = RanchorBox()
        elif args.ANCHOR_TYPE == 'KMEANS':
            self.anchors = KanchorBox()
        else:
            raise RuntimeError('Define correct anchor type')
            
        print('Cell anchors\n', self.anchors.cell_anchors)
        # pdb.set_trace()
        self.ar = self.anchors.ar
        args.ar = self.ar
        self.use_bias = True
        self.head_size = args.head_size
        self.backbone = YOLOPAFPN3DDINO()
        self.SEQ_LEN = args.SEQ_LEN
        self.HEAD_LAYERS = args.HEAD_LAYERS
        self.NUM_FEATURE_MAPS = args.NUM_FEATURE_MAPS
        
        self.reg_heads = []
        self.cls_heads = []
        self.prior_prob = 0.01
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        # for nf in range(self.NUM_FEATURE_MAPS):
        self.reg_heads = self.make_head(
            self.ar * 4, args.REG_HEAD_TIME_SIZE, self.HEAD_LAYERS)
        self.cls_heads = self.make_head_cls(
            self.ar * self.num_classes, args.CLS_HEAD_TIME_SIZE, self.HEAD_LAYERS)
        self.linearcls=dinov2linear(1536)
        self.linearup1=dinov2linearFix(1536,5632)#64 88
        self.linearup2=dinov2linearFix(1536,1408) #32 44
        self.lineardown1=dinov2linearFix(1536,352)#16 22
        self.lineardown2=dinov2linearFix(1536,88)#8 11
        self.lineardown3=dinov2linearFix(1536,24)#4 6
        
        nn.init.constant_(self.cls_heads[-1].bias, bias_value)

        if args.MODE == 'train':  # eval_iters only in test case
            self.criterion = FocalLoss(args)

    # @autocast()
    def forward(self, RGBimages ,gt_boxes=None, gt_labels=None, counts=None, img_indexs=None, get_features=False, logic=None, Cplus=None, Cminus=None, is_ulb=None):
        # sources, ego_feat = self.backbone(images)
        cls_token,sources = self.backbone(RGBimages)


        grid_sizes = [feature_map.shape[-2:] for feature_map in sources]
        ancohor_boxes = self.anchors(grid_sizes)
        
        loc = list()
        conf = list()

        cls_token=self.linearcls(cls_token)
        cls_token1=self.linearup1(cls_token)# 80 80
        cls_token2=self.linearup2(cls_token)# 40 40
        cls_token3=self.lineardown1(cls_token)# 20 20
        cls_token4=self.lineardown2(cls_token)# 10 10
        cls_token5=self.lineardown3(cls_token)# 5 5
        cls_token_list=[cls_token1,cls_token2,cls_token3,cls_token4,cls_token5]

        size_cls1 = 128
        size_cls2=176
        for i in range(len(cls_token_list)-1):
            size_cls1  //= 2
            size_cls2  //= 2
            cls_token_list[i] = cls_token_list[i].unsqueeze(1).reshape(sources[0].shape[0], 1, sources[0].shape[2], size_cls1, size_cls2)
        cls_token_list[4] = cls_token_list[4].unsqueeze(1).reshape(sources[0].shape[0], 1, sources[0].shape[2], 4, 6)
        for i,x in enumerate(sources):
            loc.append(self.reg_heads(x).permute(0, 2, 3, 4, 1).contiguous())
            conf.append(self.cls_heads(torch.cat([x,cls_token_list[i]],dim=1)).permute(0, 2, 3, 4, 1).contiguous())
        # loc b t h w c
        # conf b t h w c
        loc = torch.cat([o.view(o.size(0), o.size(1), -1) for o in loc], 2)

        conf = torch.cat([o.view(o.size(0), o.size(1), -1) for o in conf], 2)

        flat_loc = loc.view(loc.size(0), loc.size(1), -1, 4)
        flat_conf = conf.view(conf.size(0), conf.size(1), -1, self.num_classes)

        # pdb.set_trace()
        if get_features:  # testing mode with feature return
            return flat_conf, features
        elif gt_boxes is not None:  # training mode

            return self.criterion(flat_conf, flat_loc, gt_boxes, gt_labels, counts, ancohor_boxes, logic, Cplus, Cminus, is_ulb=is_ulb), is_ulb

        else:  # otherwise testing mode
            decoded_boxes = []
            for b in range(flat_loc.shape[0]):
                temp_l = []
                for s in range(flat_loc.shape[1]):
                    # torch.stack([decode(flat_loc[b], ancohor_boxes) for b in range(flat_loc.shape[0])], 0),
                    temp_l.append(decode(flat_loc[b, s], ancohor_boxes))
                decoded_boxes.append(torch.stack(temp_l, 0))
            # return torch.stack(decoded_boxes, 0), flat_conf, ego_preds
            return torch.stack(decoded_boxes, 0), flat_conf


    def make_features(self,  shared_heads):
        layers = []
        use_bias = self.use_bias
        head_size = self.head_size
        for _ in range(shared_heads):
            layers.append(nn.Conv3d(head_size, head_size, kernel_size=(
                1, 3, 3), stride=1, padding=(0, 1, 1), bias=use_bias))
            layers.append(nn.ReLU(True))

        layers = nn.Sequential(*layers)

        for m in layers.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if hasattr(m.bias, 'data'):
                    nn.init.constant_(m.bias, 0)

        return layers
    def make_head_cls(self, out_planes, time_kernel,  num_heads_layers):
        layers = []
        use_bias = self.use_bias
        head_size = self.head_size
        
        for kk in range(num_heads_layers):
       
            branch_kernel = 1
            bpad = 0
            layers.append(nn.Conv3d(head_size+1, head_size+1, kernel_size=(
                1, 3, 3), stride=1, padding=(bpad, 1, 1), bias=use_bias))
            layers.append(nn.GELU())
        tpad = time_kernel//2
        layers.append(nn.Conv3d(head_size+1, out_planes, kernel_size=(
            time_kernel, 3, 3), stride=1, padding=(tpad, 1, 1)))
        
        layers = nn.Sequential(*layers)

        for m in layers.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if hasattr(m.bias, 'data'):
                    nn.init.constant_(m.bias, 0)

        return layers
    
    def make_head(self, out_planes, time_kernel,  num_heads_layers):
        layers = []
        use_bias = self.use_bias
        head_size = self.head_size
        
        for kk in range(num_heads_layers):
            # if kk % 2 == 1 and time_kernel>1:
            #     branch_kernel = 3
            #     bpad = 1
            # else:
            branch_kernel = 1
            bpad = 0
            layers.append(nn.Conv3d(head_size, head_size, kernel_size=(
                branch_kernel, 3, 3), stride=1, padding=(bpad, 1, 1), bias=use_bias))
            layers.append(nn.GELU())

        tpad = time_kernel//2
        layers.append(nn.Conv3d(head_size, out_planes, kernel_size=(
            time_kernel, 3, 3), stride=1, padding=(tpad, 1, 1)))
        
        layers = nn.Sequential(*layers)

        for m in layers.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if hasattr(m.bias, 'data'):
                    nn.init.constant_(m.bias, 0)

        return layers    
    

def build_retinanet(args):
    if args.modelswin:
        return SwinRetinaNet(backbone_models(args), args)
    elif args.modeldino:
        return DinoRetinaNet(backbone_models(args),args)
    else:
        return RetinaNet(backbone_models(args), args)
