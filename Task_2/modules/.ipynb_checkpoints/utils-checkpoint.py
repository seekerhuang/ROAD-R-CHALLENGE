import os, sys
import shutil
import socket
import getpass
import copy
import numpy as np
from modules.box_utils import nms
import logging
import torch
import torchvision
import torchvision.ops as ops
from torchvision.ops.boxes import box_iou, distance_box_iou

# from https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
class BufferList(torch.nn.Module):
    """    
    Similar to nn.ParameterList, but for buffers
    """

    def __init__(self, buffers=None):
        super(BufferList, self).__init__()
        if buffers is not None:
            self.extend(buffers)

    def extend(self, buffers):
        offset = len(self)
        for i, buffer in enumerate(buffers):
            self.register_buffer(str(offset + i), buffer)
        return self

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.values())
        
def setup_logger(args):
    """
    Sets up the logging.
    """
    log_file_name = '{:s}/{:s}-{date:%m-%d-%H-%M-%S}_logic-{logic:s}_req-weight-{weight}.log'.format(args.SAVE_ROOT, args.MODE, date=args.DATETIME_NOW, logic=str(args.LOGIC), weight=args.req_loss_weight)
    args.log_dir = 'logs/'+args.exp_name+'/'
    # args.log_dir = args.SAVE_ROOT+'/logs/'+args.exp_name+'/'
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)
        
    added_log_file = '{}{}-{date:%m-%d-%Hx}.log'.format(args.log_dir, args.MODE, date=args.DATETIME_NOW)

   
    # Set up logging format.
    _FORMAT = "[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s"

    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO, format=_FORMAT, stream=sys.stdout
    )
    logging.getLogger().addHandler(logging.FileHandler(log_file_name, mode='a'))
    # logging.getLogger().addHandler(logging.FileHandler(added_log_file, mode='a'))


def get_logger(name):
    """
    Retrieve the logger with the specified name or, if name is None, return a
    logger which is the root logger of the hierarchy.
    Args:
        name (string): name of the logger.
    """
    return logging.getLogger(name)

def copy_source(source_dir):
    if not os.path.isdir(source_dir):
        os.system('mkdir -p ' + source_dir)
    
    for dirpath, dirs, files in os.walk('./', topdown=True):
        for file in files:
            if file.endswith('.py'): #fnmatch.filter(files, filepattern):
                shutil.copy2(os.path.join(dirpath, file), source_dir)


def set_args(args):
    args.MAX_SIZE = int(args.MIN_SIZE*1.35)
    args.MILESTONES = [int(val) for val in args.MILESTONES.split(',')]
    #args.GAMMAS = [float(val) for val in args.GAMMAS.split(',')]
    args.EVAL_EPOCHS = [int(val) for val in args.EVAL_EPOCHS.split(',')]

    args.TRAIN_SUBSETS = [val for val in args.TRAIN_SUBSETS.split(',') if len(val)>1]
    args.VAL_SUBSETS = [val for val in args.VAL_SUBSETS.split(',') if len(val)>1]
    args.TEST_SUBSETS = [val for val in args.TEST_SUBSETS.split(',') if len(val)>1]
    args.TUBES_EVAL_THRESHS = [ float(val) for val in args.TUBES_EVAL_THRESHS.split(',') if len(val)>0.0001]
    args.model_subtype = args.MODEL_TYPE.split('-')[0]
    ## check if subsets are okay
    possible_subets = ['test', 'train','val']
    for idx in range(1,4):
        possible_subets.append('train_'+str(idx))        
        possible_subets.append('val_'+str(idx))        

    if len(args.VAL_SUBSETS) < 1 and args.DATASET == 'road':
        args.VAL_SUBSETS = [ss.replace('train', 'val') for ss in args.TRAIN_SUBSETS]
    if len(args.TEST_SUBSETS) < 1:
        # args.TEST_SUBSETS = [ss.replace('train', 'val') for ss in args.TRAIN_SUBSETS]
        args.TEST_SUBSETS = args.VAL_SUBSETS
    
    for subsets in [args.TRAIN_SUBSETS, args.VAL_SUBSETS, args.TEST_SUBSETS]:
        for subset in subsets:
            assert subset in possible_subets, 'subest should from one of these '+''.join(possible_subets)

    args.DATASET = args.DATASET.lower()
    args.ARCH = args.ARCH.lower()

    args.MEANS =[0.485, 0.456, 0.406]
    args.STDS = [0.229, 0.224, 0.225]

    username = getpass.getuser()
    hostname = socket.gethostname()
    args.hostname = hostname
    args.user = username
    
    args.model_init = 'kinetics'

    args.MODEL_PATH = args.MODEL_PATH[:-1] if args.MODEL_PATH.endswith('/') else args.MODEL_PATH 

    assert args.MODEL_PATH.endswith('kinetics-pt') or args.MODEL_PATH.endswith('imagenet-pt') 
    args.model_init = 'imagenet' if args.MODEL_PATH.endswith('imagenet-pt') else 'kinetics'
    
    if args.MODEL_PATH == 'imagenet':
        args.MODEL_PATH = os.path.join(args.MODEL_PATH, args.ARCH+'.pth')
    else:
        args.MODEL_PATH = os.path.join(args.MODEL_PATH, args.ARCH+args.MODEL_TYPE+'.pth')
            
    
    print('Your working directories are::\nLOAD::> ', args.DATA_ROOT, '\nSAVE::> ', args.SAVE_ROOT)
    if args.RESUME>0:
        # print('Your model will be initialized using checkpoint from epoch', str(args.RESUME), 'from', args.DATA_ROOT)
        print('Your model will be initialized using', '{:s}/model_{:06d}.pth'.format(args.SAVE_ROOT, args.RESUME))
    else:
        print('Your model will be initialized using', args.MODEL_PATH)

    return args


def create_exp_name(args):
    """Create name of experiment using training parameters """
    splits = ''.join([split[0]+split[-1] for split in args.TRAIN_SUBSETS])
    args.exp_name = '{:s}{:s}{:d}-P{:s}-b{:0d}s{:d}x{:d}x{:d}-{:s}{:s}-h{:d}x{:d}x{:d}-{date:%m-%d-%H-%M-%Sx}'.format(
        args.ARCH, args.MODEL_TYPE,
        args.MIN_SIZE, args.model_init, args.BATCH_SIZE,
        args.SEQ_LEN, args.MIN_SEQ_STEP, args.MAX_SEQ_STEP,
        args.DATASET, splits,
        args.HEAD_LAYERS, args.CLS_HEAD_TIME_SIZE,
        args.REG_HEAD_TIME_SIZE,
        date=args.DATETIME_NOW,
    )

    if (args.MODE == 'gen_dets') or (args.RESUME>0):
        if len(args.EXP_NAME) > 0:
            args.exp_name = args.EXP_NAME
            print('Do not create a new experiment dir, instead use', args.exp_name)
        else:
            raise Exception('if MODE is gen_dets or if RESUME>0, then most provide EXP_NAME')

    if len(args.EXP_NAME) > 0:
        args.SAVE_ROOT = args.EXP_NAME
    else:
        args.SAVE_ROOT += args.DATASET+'/'
        # args.SAVE_ROOT = args.SAVE_ROOT+'cache/'+args.exp_name+'/'
        # "log-lo_cache_logic_None_0.0"
        # args.SAVE_ROOT = args.SAVE_ROOT + 'log-lo_cache_logic' + str(args.LOGIC) + '_' + str(
        #     args.req_loss_weight) + '/' + args.exp_name + '/'
        args.SAVE_ROOT = args.SAVE_ROOT + 'logic-ssl_cache_' + str(args.LOGIC) + '_' + str(
            args.req_loss_weight) +   '/' + args.exp_name + '/'

        if not os.path.isdir(args.SAVE_ROOT):
            print('Create: ', args.SAVE_ROOT)
            os.makedirs(args.SAVE_ROOT)

    return args
    
# Freeze batch normlisation layers
def set_bn_eval(m):
    classname = m.__class__.__name__
    if (classname.find('BatchNorm') > -1) or (classname.find('SyncBatchNorm') > -1):
        m.eval()
        if ('upcovhead' in classname) or ('tconv' in classname) or ('conv2' in classname) or ('deconv3' in classname) or ('uptime' in classname):
            m.eval()
            if m.affine:
                      m.weight.requires_grad = False
                      m.bias.requires_grad = False
        else:
            m.eval()
            if m.affine:
                m.weight.requires_grad = False
                m.bias.requires_grad = False
        

def get_individual_labels(gt_boxes, tgt_labels):
    # print(gt_boxes.shape, tgt_labels.shape)
    new_gts = np.zeros((gt_boxes.shape[0]*20, 5))
    ccc = 0
    for n in range(tgt_labels.shape[0]):
        for t in range(tgt_labels.shape[1]):
            if tgt_labels[n,t]>0:
                new_gts[ccc, :4] = gt_boxes[n,:]
                new_gts[ccc, 4] = t
                ccc += 1
    return new_gts[:ccc,:]


def get_individual_location_labels(gt_boxes, tgt_labels):
    return [gt_boxes, tgt_labels]


def filter_detections(args, scores, decoded_boxes_batch):
    c_mask = scores.gt(args.CONF_THRESH)  # greater than minmum threshold
    scores = scores[c_mask].squeeze()
    if scores.dim() == 0 or scores.shape[0] == 0:
        return np.asarray([])
    
    boxes = decoded_boxes_batch[c_mask, :].view(-1, 4)
    ids, counts = nms(boxes, scores, args.NMS_THRESH, args.TOPK*5)  # idsn - ids after nms
    scores = scores[ids[:min(args.TOPK,counts)]].cpu().numpy()
    boxes = boxes[ids[:min(args.TOPK,counts)]].cpu().numpy()
    cls_dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=True)

    return cls_dets


def diou_nms(boxes, scores, iou_threshold=0.5, diou_threshold=0.5):
    """
    Performs dIoU-NMS on the input boxes.
    Args:
        boxes (Tensor[N, 4]): input boxes in (x1, y1, x2, y2) format.
        scores (Tensor[N]): input box scores.
        iou_threshold (float): IoU threshold for regular NMS.
        diou_threshold (float): dIoU threshold for dIoU-NMS.
    Returns:
        indices (Tensor[K]): indices of the boxes to keep after NMS.
    """
    # Sort boxes by scores in descending order
    sorted_idx = torch.argsort(scores, descending=True)

    # Initialize list of indices to keep
    keep = []

    while len(sorted_idx) > 0:
        # Pick the box with the highest score
        idx = sorted_idx[0]
        keep.append(idx)

        # Compute IoU between the picked box and the remaining boxes
        ious = box_iou(boxes[idx].unsqueeze(0), boxes[sorted_idx])

        # Compute dIoU between the picked box and the remaining boxes
        dious = distance_box_iou(boxes[idx].unsqueeze(0), boxes[sorted_idx])

        # Find the indices of the boxes with IoU greater than the threshold
        iou_mask = ious.squeeze(0) > iou_threshold

        # Find the indices of the boxes with dIoU greater than the threshold
        diou_mask = dious.squeeze(0) > diou_threshold

        # Apply NMS based on both IoU and dIoU
        mask = iou_mask & diou_mask

        # Remove the indices of the boxes that overlap too much with the picked box
        sorted_idx = sorted_idx[~mask]

    # Return the list of indices to keep
    return torch.tensor(keep, dtype=torch.long)





def soft_nms_pytorch(dets, box_scores, sigma=0.5, thresh=0.001, cuda=0):
    """
    Build a pytorch implement of Soft NMS algorithm.
    # Augments
        dets:        boxes coordinate tensor (format:[y1, x1, y2, x2])
        box_scores:  box score tensors
        sigma:       variance of Gaussian function
        thresh:      score thresh
        cuda:        CUDA flag
    # Return
        the index of the selected boxes
    """

    # Indexes concatenate boxes with the last column
    N = dets.shape[0]
    if cuda:
        indexes = torch.arange(0, N, dtype=torch.float).cuda().view(N, 1)
    else:
        indexes = torch.arange(0, N, dtype=torch.float).view(N, 1)
        dets=dets.cpu()
        box_scores=box_scores.cpu()
    dets = torch.cat((dets, indexes), dim=1)

    # The order of boxes coordinate is [y1,x1,y2,x2]
    y1 = dets[:, 0]
    x1 = dets[:, 1]
    y2 = dets[:, 2]
    x2 = dets[:, 3]
    scores = box_scores
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    for i in range(N):
        # intermediate parameters for later parameters exchange
        tscore = scores[i].clone()
        pos = i + 1

        if i != N - 1:
            maxscore, maxpos = torch.max(scores[pos:], dim=0)
            if tscore < maxscore:
                dets[i], dets[maxpos.item() + i + 1] = dets[maxpos.item() + i + 1].clone(), dets[i].clone()
                scores[i], scores[maxpos.item() + i + 1] = scores[maxpos.item() + i + 1].clone(), scores[i].clone()
                areas[i], areas[maxpos + i + 1] = areas[maxpos + i + 1].clone(), areas[i].clone()

        # IoU calculate
        yy1 = np.maximum(dets[i, 0].to("cpu").numpy(), dets[pos:, 0].to("cpu").numpy())
        xx1 = np.maximum(dets[i, 1].to("cpu").numpy(), dets[pos:, 1].to("cpu").numpy())
        yy2 = np.minimum(dets[i, 2].to("cpu").numpy(), dets[pos:, 2].to("cpu").numpy())
        xx2 = np.minimum(dets[i, 3].to("cpu").numpy(), dets[pos:, 3].to("cpu").numpy())
        
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = torch.tensor(w * h).cuda() if cuda else torch.tensor(w * h)
        ovr = torch.div(inter, (areas[i] + areas[pos:] - inter))

        # Gaussian decay
        weight = torch.exp(-(ovr * ovr) / sigma)
        scores[pos:] = weight * scores[pos:]

    # select the boxes and keep the corresponding indexes
    keep = dets[:, 4][scores > thresh].int()
    # sorted_idx = torch.argsort(box_scores, descending=True)
    # keep = keep[sorted_idx]
    # box_scores = box_scores[sorted_idx]
    # dets = dets[sorted_idx, :]

    return keep   





def filter_detections_for_tubing(args, scores, decoded_boxes_batch, confidences):
    c_mask = scores.gt(args.CONF_THRESH)  # greater than minmum threshold
    scores = scores[c_mask].squeeze()
    if scores.dim() == 0 or scores.shape[0] == 0:
        return  np.zeros((0,200))
    
    boxes = decoded_boxes_batch[c_mask, :].clone().view(-1, 4)
    numc = confidences.shape[-1]
    confidences = confidences[c_mask,:].clone().view(-1, numc)

    max_k = min(args.TOPK*60, scores.shape[0])
    ids, counts = nms(boxes, scores, args.NMS_THRESH, max_k)  # idsn - ids after nms
    scores = scores[ids[:min(args.TOPK,counts)]].cpu().numpy()
    boxes = boxes[ids[:min(args.TOPK,counts)],:].cpu().numpy()
    confidences = confidences[ids[:min(args.TOPK, counts)],:].cpu().numpy()
    cls_dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=True)
    save_data = np.hstack((cls_dets, confidences[:,1:])).astype(np.float32)
    #print(save_data.shape)
    return save_data


def filter_detections_for_dumping(args, scores, decoded_boxes_batch, confidences):
    c_mask = scores.gt(args.GEN_CONF_THRESH)  # greater than minmum threshold
    scores = scores[c_mask].squeeze()
    if scores.dim() == 0 or scores.shape[0] == 0:
        return np.zeros((0,5)), np.zeros((0,200))
    
    boxes = decoded_boxes_batch[c_mask, :].clone().view(-1, 4)
    numc = confidences.shape[-1]
    confidences = confidences[c_mask,:].clone().view(-1, numc)

    # sorted_ind = np.argsort(-scores.cpu().numpy())
    # sorted_ind = sorted_ind[:topk*10]
    # boxes_np = boxes.cpu().numpy()
    # confidences_np = confidences.cpu().numpy()
    # save_data = np.hstack((boxes_np[sorted_ind,:], confidences_np[sorted_ind, :]))
    # args.GEN_TOPK, args.GEN_NMS
     
    max_k = min(args.GEN_TOPK*500, scores.shape[0])
    # ids, counts = nms(boxes, scores, args.GEN_NMS, max_k)  # idsn - ids after nms
    # scores = scores[ids[:min(args.GEN_TOPK,counts)]].cpu().numpy()
    # boxes = boxes[ids[:min(args.GEN_TOPK,counts)],:].cpu().numpy()
    # confidences = confidences[ids[:min(args.GEN_TOPK, counts)],:].cpu().numpy()
    # print("confi",confidences)
#     keep = ops.batched_nms(boxes, scores, confidences[:, 1:], args.GEN_NMS)
#     keep = keep[:min(args.GEN_TOPK, len(keep))]

#     scores = scores[keep].cpu().numpy()
#     boxes = boxes[keep, :].cpu().numpy()
    # confidences = confidences[keep, :].cpu().numpy()
    # boxes.cpu().numpy()
    keep = diou_nms(boxes, scores, iou_threshold=args.GEN_NMS, diou_threshold=0.46) 
    # scores_np = scores[keep].detach().cpu().numpy()
    # order = np.argsort(scores_np)
    # scores_np = scores_np[order[::-1]]
    # keep = keep.sort(0)
  
  
    
    # scores=scores.cpu().numpy()
    # keep = keep[scores[keep].argsort()[::-1]]
    keep = keep[:min(args.GEN_TOPK, len(keep))]
    
    scores = scores[keep].cpu().numpy()
    boxes = boxes[keep,:].cpu().numpy()
    confidences = confidences[keep,:].cpu().numpy()
    
    # sorted_ind = np.argsort(-scores)
    # scores = scores[sorted_ind]
    # boxes = boxes[sorted_ind]
    # confidences = confidences[sorted_ind]
#     keep = torchvision.ops.nms(boxes, scores, args.GEN_NMS)
#     keep = keep[:min(args.GEN_TOPK, len(keep))]
#     # print("batchnms")

#     scores = scores[keep].cpu().numpy()
#     boxes = boxes[keep, :].cpu().numpy()
#     confidences = confidences[keep, :].cpu().numpy()
    cls_dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=True)
    save_data = np.hstack((cls_dets, confidences[:,1:])).astype(np.float32)
    #print(save_data.shape)
    return cls_dets, save_data

def make_joint_probs_from_marginals(frame_dets, childs, num_classes_list, start_id=4):
    
    # pdb.set_trace()

    add_list = copy.deepcopy(num_classes_list[:3])
    add_list[0] = start_id+1
    add_list[1] = add_list[0]+add_list[1]
    add_list[2] = add_list[1]+add_list[2]
    # for ind in range(frame_dets.shape[0]):
    for nlt, ltype in enumerate(['duplex','triplet']):
        lchilds = childs[ltype+'_childs']
        lstart = start_id
        for num in num_classes_list[:4+nlt]:
            lstart += num
        
        for c in range(num_classes_list[4+nlt]):
            tmp_scores = []
            for chid, ch in enumerate(lchilds[c]):
                if len(tmp_scores)<1:
                    tmp_scores = copy.deepcopy(frame_dets[:,add_list[chid]+ch])
                else:
                    tmp_scores *= frame_dets[:,add_list[chid]+ch]
            frame_dets[:,lstart+c] = tmp_scores

    return frame_dets



def eval_strings():
    return ["Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = ",
            "Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = ",
            "Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = ",    
            "Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = ",
            "Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = ",
            "Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = ",
            "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = ",
            "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = ",
            "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = ",
            "Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = ",
            "Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = ",
            "Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = "]
