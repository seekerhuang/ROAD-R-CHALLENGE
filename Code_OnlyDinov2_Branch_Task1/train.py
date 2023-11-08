
import math
import time

import torch
import torch.utils.data as data_utils

from data import custom_collate
from modules import AverageMeter
from modules import utils
from modules.solver import get_optim
from utils_ssl import compute_req_matrices
from val import validate
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
import numpy as np
import torch.nn as nn
import collections
import torch.optim.lr_scheduler as lr_scheduler
logger = utils.get_logger(__name__)


def get_time_dim(net):
    for name, param in net.keys():
        if 'conv' in name and param.dim() == 5:
            return param.size(2)
    return None

def inflate_weight(weight,time_dim):
    for k in weight.keys():
        if ('conv' in k) and ('bn' not in k) and ('lateral_conv0' not in k) and ('reduce_conv1' not in k) and ('C3_p4' not in k) and ('C3_n4' not in k):
            w = weight[k].detach().cpu().numpy()
            w = np.expand_dims(w, axis=2)
            target_shape = list(w.shape) 
            target_shape[2] = time_dim
            w_expanded = np.broadcast_to(w, target_shape)/time_dim
            weight[k]=torch.from_numpy(w_expanded)
        if ('lateral_conv0' in k) and ('bn' not in k):
            w = weight[k].detach().cpu().numpy()
            w = np.expand_dims(w, axis=2)
            target_shape = list(w.shape) 
            target_shape[2] = time_dim*3
            w_expanded = np.broadcast_to(w, target_shape)/(time_dim*3)
            weight[k]=torch.from_numpy(w_expanded)
        if ('reduce_conv1' in k) and ('bn' not in k):
            w = weight[k].detach().cpu().numpy()
            w = np.expand_dims(w, axis=2)
            target_shape = list(w.shape) 
            target_shape[2] = time_dim*3
            w_expanded = np.broadcast_to(w, target_shape)/(time_dim*3)
            weight[k]=torch.from_numpy(w_expanded)
        if ('C3_p4' in k) and ('bn' not in k):
            w = weight[k].detach().cpu().numpy()
            w = np.expand_dims(w, axis=2)
            target_shape = list(w.shape) 
            target_shape[2] = time_dim*3
            w_expanded = np.broadcast_to(w, target_shape)/(time_dim*3)
            weight[k]=torch.from_numpy(w_expanded)
        if ('C3_n4' in k) and ('bn' not in k):
            w = weight[k].detach().cpu().numpy()
            w = np.expand_dims(w, axis=2)
            target_shape = list(w.shape) 
            target_shape[2] = time_dim*3
            w_expanded = np.broadcast_to(w, target_shape)/(time_dim*3)
            weight[k]=torch.from_numpy(w_expanded)
    return weight


def inflate_weight_channel(weight, num_channels):

    for k in weight.keys():
        if ('cls_heads' in k) and ('cls_heads.6' not in k) and ('.bias' not in k):
            w = weight[k].detach().cpu().numpy()
            weight_shape = w.shape
            inflated_shape = (num_channels + 1, num_channels + 1) + weight_shape[2:]
            print(inflated_shape)

            inflated_weight = np.zeros(inflated_shape)
            inflated_weight[:num_channels, :num_channels] = w
            average_channel = np.mean(w, axis=0)
            print("avg",average_channel.shape)

            inflated_weight[num_channels, :num_channels] = average_channel
            inflated_weight[:num_channels, num_channels] = average_channel
            inflated_weight[num_channels, num_channels] = np.mean(average_channel)

            print(inflated_weight.shape)
            weight[k] = torch.from_numpy(inflated_weight)
        if ('cls_heads' in k) and ('cls_heads.6' not in k) and ('.bias' in k):
            bias = weight[k].detach().cpu().numpy()
            inflated_bias = np.concatenate([bias, [0]]) 
            weight[k] = torch.from_numpy(inflated_bias)
            
    return weight


def setup_training(args, net):
    optimizer, _ , solver_print_str = get_optim(args, net)
    num_steps = 160 * args.MAX_EPOCHS# !!! Note: if you change the number of bs, you also need to change this number !!!
    scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                          first_cycle_steps=num_steps,
                                          cycle_mult=1.0,
                                          max_lr=args.LR,
                                          min_lr=1e-6,
                                          warmup_steps=800,
                                          gamma=0.5)
    if args.TENSORBOARD:
        from tensorboardX import SummaryWriter
    source_dir = args.SAVE_ROOT + 'source/'  # where to save the source
    # utils.copy_source(source_dir)
    args.START_EPOCH = 1
    if args.RESUME > 0:
        # raise Exception('Not implemented')
        args.START_EPOCH = args.RESUME + 1
        # args.iteration = args.START_EPOCH
        for _ in range(args.RESUME):
            scheduler.step()
        model_file_name = '{:s}/model_{:06d}.pth'.format(args.SAVE_ROOT, args.RESUME)
        optimizer_file_name = '{:s}/optimizer_{:06d}.pth'.format(args.SAVE_ROOT, args.RESUME)
        # sechdular_file_name = '{:s}/optimizer_{:06d}.pth'.format(args.SAVE_ROOT, args.START_EPOCH)
        net.load_state_dict(torch.load(model_file_name))
        optimizer.load_state_dict(torch.load(optimizer_file_name))
        lrt=[]
        for param_group in optimizer.param_groups:
                lrt.append(param_group['lr'])
        logger.info('After loading checkpoint from epoch {:}, the learning rate is {:}'.format(args.RESUME, lrt))
    if args.TENSORBOARD:
        log_dir = '{:s}/log-lo_tboard-{}-{date:%m-%d-%H-%M-%S}_logic-{logic:s}_req-weight-{weight}'.format(args.log_dir,
                                                                                                           args.MODE,
                                                                                                           date=args.DATETIME_NOW,
                                                                                                           logic=str(
                                                                                                               args.LOGIC),
                                                                                                           weight=args.req_loss_weight)
        # log_dir = '{:s}/log-lo_tboard-{}_logic-{logic:s}_req-weight-{weight}'.format(args.log_dir, args.MODE, logic=str(args.LOGIC), weight=args.req_loss_weight)
        args.sw = SummaryWriter(log_dir)
        logger.info('Created tensorboard log dir ' + log_dir)

    if args.pretrained_model_path is not None and args.RESUME == 0:

        checkpoint = torch.load(args.pretrained_model_path)
        checkpoint2=torch.load(args.pretrained_model_path2)
        checkpoint_fpn=torch.load(args.pretrained_model_pathfpn)
        checkpoint_fpn=collections.OrderedDict(checkpoint_fpn['model'])
        backbone_stat_dict_fpn = {}
        for k in list(checkpoint_fpn.keys()):
            
            # 修改层名
            if k.startswith('backbone.lateral_conv0'):
                backbone_stat_dict_fpn[k.replace('backbone.lateral_conv0', 'backbone.lateral_conv0')] = checkpoint_fpn[k]
            # 修改层名
            if k.startswith('backbone.C3_p4'):    
                backbone_stat_dict_fpn[k.replace('backbone.C3_p4', 'backbone.C3_p4')] = checkpoint_fpn[k]

            if k.startswith('backbone.reduce_conv1'):
                backbone_stat_dict_fpn[k.replace('backbone.reduce_conv1', 'backbone.reduce_conv1')] = checkpoint_fpn[k]
            
            if k.startswith('backbone.C3_p3'):
                backbone_stat_dict_fpn[k.replace('backbone.C3_p3', 'backbone.C3_p3')] = checkpoint_fpn[k]
            
            if k.startswith('backbone.bu_conv2'):
                backbone_stat_dict_fpn[k.replace('backbone.bu_conv2', 'backbone.bu_conv2')] = checkpoint_fpn[k]
                
            if k.startswith('backbone.C3_n3'):
                backbone_stat_dict_fpn[k.replace('backbone.C3_n3', 'backbone.C3_n3')] = checkpoint_fpn[k]
            
            if k.startswith('backbone.bu_conv1'):
                backbone_stat_dict_fpn[k.replace('backbone.bu_conv1', 'backbone.bu_conv1')] = checkpoint_fpn[k]
                
            if k.startswith('backbone.C3_n4'):
                backbone_stat_dict_fpn[k.replace('backbone.C3_n4', 'backbone.C3_n4')] = checkpoint_fpn[k]

        backbone_stat_dict_fpn=inflate_weight(backbone_stat_dict_fpn,1)

        backbone_stat_dict2 = {}
        for k in checkpoint2.keys():
            
            if ('heads' in k):
                
                if k.startswith('module.backbone'):
                    backbone_stat_dict2[k.replace('module.', '')] = checkpoint2[k]
                if k.startswith('module.reg_heads'):
                    backbone_stat_dict2[k.replace('module.', '')] = checkpoint2[k]
                if k.startswith('module.cls_heads') and ('cls_heads.6' not in k):
                    backbone_stat_dict2[k.replace('module.', '')] = checkpoint2[k]

        new_stat_dict={}
        backbone_stat_dict2=inflate_weight_channel(backbone_stat_dict2,256)
        for k in backbone_stat_dict2.keys():
            if 'cls_heads' in k:
                print(k,backbone_stat_dict2[k].shape)

        new_stat_dict.update(backbone_stat_dict2)
        new_stat_dict.update(backbone_stat_dict_fpn)
        net.load_state_dict(new_stat_dict,strict=False)
        result=net.load_state_dict(new_stat_dict,strict=False)
        
        print("missing")
        print(result.missing_keys) 
        print("unexpcted")
        print(result.unexpected_keys) 
        logger.info("Load pretrained model {:}".format(args.pretrained_model_path))
    if args.MULTI_GPUS:
            logger.info('\nLets do dataparallel\n')
            net = torch.nn.DataParallel(net)
            net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    logger.info(str(net))
    logger.info(solver_print_str)

    logger.info('EXPERIMENT NAME:: ' + args.exp_name)
    logger.info('Training FPN with {} + {} as backbone '.format(args.ARCH, args.MODEL_TYPE))
    return args, optimizer,scheduler,net

Init_Epoch = 0
Freeze_Epoch = 10
# Unfreeze_Epoch =100

def train(args, net, train_dataset, val_dataset):
    epoch_size = len(train_dataset) // args.BATCH_SIZE
    args.MAX_ITERS = epoch_size #args.MAX_EPOCHS * epoch_size
    args, optimizer,scheduler,net = setup_training(args, net)

    train_data_loader = data_utils.DataLoader(train_dataset, args.BATCH_SIZE, num_workers=args.NUM_WORKERS,
                                              shuffle=True, pin_memory=True, collate_fn=custom_collate, drop_last=True)

    val_data_loader = data_utils.DataLoader(val_dataset, args.BATCH_SIZE, num_workers=args.NUM_WORKERS,
                                            shuffle=False, pin_memory=True, collate_fn=custom_collate)

    # TO REMOVE
    # if not args.DEBUG_num_iter:
    #     net.eval()
    #     run_val(args, val_data_loader, val_dataset, net, 0, 0)

    iteration = 0
    for epoch in range(args.START_EPOCH, args.MAX_EPOCHS + 1):
        if epoch == 10:
            for param_group in optimizer.param_groups:
                param_group['weight_decay'] = 0.038
        if epoch == 12:
            for param_group in optimizer.param_groups:
                param_group['weight_decay'] = 0.04
        if epoch == 14:
            for param_group in optimizer.param_groups:
                param_group['weight_decay'] = 0.042
        if epoch==16:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr']/2.6
                param_group['weight_decay'] = 0.04
        if epoch == 18:
            for param_group in optimizer.param_groups:
                param_group['weight_decay'] = 0.04
        if epoch == 20:
            for param_group in optimizer.param_groups:
                param_group['weight_decay'] = 0.04
        if epoch == 21:
            for param_group in optimizer.param_groups:
                param_group['weight_decay'] = 0.04
        if epoch==22:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 1.6
                param_group['weight_decay'] = 0.04
            # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=(args.MAX_EPOCHS-epoch)*524, eta_min=1e-9)  
            # scheduler.last_epoch = 21
        # if epoch==30:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] /= 1.4
        print('LR at epoch {:} is {:}'.format(epoch, scheduler.get_lr()))
        net.train()

        if args.FBN:
            if args.MULTI_GPUS:
                net.module.backbone.backbone.apply(utils.set_bn_eval)
                net.module.backbone.backbone.rgbbackbone.apply(utils.set_bn_eval)
                net.module.backbone.backbone.rgbbackbone.dinobackbone.apply(utils.set_bn_eval)
                net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
            else:
                net.backbone.apply(utils.set_bn_eval)
        iteration = run_train(args, train_data_loader, net, optimizer, epoch, iteration,scheduler)
        if epoch % args.VAL_STEP == 0 or epoch == args.MAX_EPOCHS:
            net.eval()
            run_val(args, val_data_loader, val_dataset, net, epoch, iteration)

        # scheduler.step()

def run_train(args, train_data_loader, net, optimizer, epoch, iteration,scheduler):
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = {'gt': AverageMeter(), 'ulb': AverageMeter(), 'all': AverageMeter()}
    loc_losses = {'gt': AverageMeter(), 'ulb': AverageMeter(), 'all': AverageMeter()}
    cls_losses = {'gt': AverageMeter(), 'ulb': AverageMeter(), 'all': AverageMeter()}
    req_losses = {'gt': AverageMeter(), 'ulb': AverageMeter(), 'all': AverageMeter()}

    torch.cuda.synchronize()
    start = time.perf_counter()

    if args.LOGIC is not None:
        Cplus, Cminus = compute_req_matrices(args)

    # for internel_iter, (images, gt_boxes, gt_labels, ego_labels, counts, img_indexs, wh) in enumerate(train_data_loader):
    for internel_iter, (mix_images, mix_gt_boxes, mix_gt_labels, mix_counts, mix_img_indexs, mix_wh, _, _, mix_is_ulb) in enumerate(train_data_loader):
        if args.DEBUG_num_iter and internel_iter > 22:
            logger.info('DID 5 ITERATIONS IN TRAIN, break.... for debugging only')
            break

        images = mix_images.cuda(0, non_blocking=True)
        # flows=flows=mix_flows.cuda(0,non_blocking=True)
        gt_boxes = mix_gt_boxes.cuda(0, non_blocking=True)
        gt_labels = mix_gt_labels.cuda(0, non_blocking=True)
        counts = mix_counts.cuda(0, non_blocking=True)
        img_indexs = mix_img_indexs

        loc_loss_dict = {'gt': [], 'ulb': [], 'all': []}
        conf_loss_dict = {'gt': [], 'ulb': [], 'all': []}
        req_loss_dict = {'gt': [], 'ulb': [], 'all': []}

        iteration += 1
        if args.LOGIC is not None:
            Cplus = Cplus.cuda(0, non_blocking=True)
            Cminus = Cminus.cuda(0, non_blocking=True)
        data_time.update(time.perf_counter() - start)


        mix_mask_is_ulb = torch.tensor([all(mix_is_ulb[b]) for b in range(mix_images.shape[0])])

        # forward
        torch.cuda.synchronize()

        # print(images.size(), anchors.size())
        optimizer.zero_grad()
        # pdb.set_trace()

        if args.LOGIC is None:
            (mix_loc_loss, mix_conf_loss), selected_is_ulb = net(images, gt_boxes, gt_labels, counts, img_indexs, is_ulb=mix_mask_is_ulb)
            # Mean over the losses computed on the different GPUs
            loc_loss, conf_loss = mix_loc_loss.mean(), mix_conf_loss.mean()
            loss = loc_loss + conf_loss
        else:
            (mix_loc_loss, mix_conf_loss, mix_req_loss), selected_is_ulb = net(images, gt_boxes, gt_labels, counts, img_indexs, logic=args.LOGIC, Cplus=Cplus, Cminus=Cminus, is_ulb=mix_mask_is_ulb)
            # Mean over the losses computed on the different GPUs
            loc_loss, conf_loss, req_loss = mix_loc_loss.mean(), mix_conf_loss.mean(), mix_req_loss.mean()
            loss = loc_loss + conf_loss + args.req_loss_weight * req_loss

        if args.log_ulb_gt_separately:  # for DataParallel only
            for i, elem in enumerate(selected_is_ulb):
                type_elem = 'ulb' if elem else 'gt'
                loc_loss_dict[type_elem].append(mix_loc_loss[i])  # for DataParallel only
                conf_loss_dict[type_elem].append(mix_conf_loss[i])  # for DataParallel only
                if args.LOGIC is not None:
                    req_loss_dict[type_elem].append(mix_req_loss[i])
        else:
            type_elem = 'all'
            loc_loss_dict[type_elem].append(mix_loc_loss.mean())
            conf_loss_dict[type_elem].append(mix_conf_loss.mean())
            if args.LOGIC is not None:
                req_loss_dict[type_elem].append(mix_req_loss.mean())

        loss.backward()
        optimizer.step()
        scheduler.step()
        
        for elem in mix_mask_is_ulb.unique():
            if args.log_ulb_gt_separately:  # for DataParallel only
                type_elem = 'ulb' if elem else 'gt'
            else:
                type_elem = 'all'

            loc_loss_dict[type_elem] = torch.tensor(loc_loss_dict[type_elem]).mean()
            conf_loss_dict[type_elem] = torch.tensor(conf_loss_dict[type_elem]).mean()

            loc_loss = loc_loss_dict[type_elem].item()
            conf_loss = conf_loss_dict[type_elem].item()
            if math.isnan(loc_loss) or loc_loss > 300:
                lline = '\n\n\n We got faulty LOCATION loss {} {} {}\n\n\n'.format(loc_loss, conf_loss, type_elem)
                logger.info(lline)
                loc_loss = 20.0
            if math.isnan(conf_loss) or conf_loss > 300:
                lline = '\n\n\n We got faulty CLASSIFICATION loss {} {} {}\n\n\n'.format(loc_loss, conf_loss, type_elem)
                logger.info(lline)
                conf_loss = 20.0

            loc_losses[type_elem].update(loc_loss)
            cls_losses[type_elem].update(conf_loss)
            if args.LOGIC is None:
                losses[type_elem].update(loc_loss + conf_loss)
            else:
                req_loss_dict[type_elem] = torch.tensor(req_loss_dict[type_elem]).mean()
                req_loss = req_loss_dict[type_elem]
                req_losses[type_elem].update(req_loss)
                losses[type_elem].update(loc_loss + conf_loss + req_loss)  # do not multiply by req weight, so exp are comparable

            if internel_iter % args.LOG_STEP == 0 and iteration > args.LOG_START and internel_iter > 0:
                if args.TENSORBOARD:
                    loss_group = dict()
                    loss_group['Classification-'+type_elem] = cls_losses[type_elem].val
                    loss_group['Localisation-'+type_elem] = loc_losses[type_elem].val
                    loss_group['Requirements-'+type_elem] = req_losses[type_elem].val
                    loss_group['Overall-'+type_elem] = losses[type_elem].val
                    args.sw.add_scalars('Losses', loss_group, iteration)
                    args.sw.add_scalar('LearningRate', scheduler.get_lr()[0], iteration)
                    args.sw.add_scalars('Losses_ep', loss_group, epoch)

                print_line = 'Iteration [{:d}/{:d}]{:06d}/{:06d} losses for {:} loc-loss {:.2f}({:.2f}) cls-loss {:.2f}({:.2f}) req-loss {:.8f}({:.8f}) ' \
                             'overall-loss {:.2f}({:.2f})'.format(epoch,
                                                                  args.MAX_EPOCHS, internel_iter, args.MAX_ITERS, type_elem,
                                                                  loc_losses[type_elem].val, loc_losses[type_elem].avg,
                                                                  cls_losses[type_elem].val,
                                                                  cls_losses[type_elem].avg, req_losses[type_elem].val,
                                                                  req_losses[type_elem].avg, losses[type_elem].val,
                                                                  losses[type_elem].avg)
                logger.info(print_line)
            if type_elem == 'all':
                break

        torch.cuda.synchronize()
        batch_time.update(time.perf_counter() - start)
        start = time.perf_counter()

        if internel_iter % args.LOG_STEP == 0 and iteration > args.LOG_START and internel_iter>0:
            print_line = 'DataTime {:0.2f}({:0.2f}) Timer {:0.2f}({:0.2f})'.format(10 * data_time.val, 10 * data_time.avg, 10 * batch_time.val, 10 * batch_time.avg)
            logger.info(print_line)

            if internel_iter % (args.LOG_STEP*20) == 0:
                logger.info(args.exp_name)
    logger.info('Saving state, epoch:' + str(epoch))
    torch.save(net.state_dict(), '{:s}/model_{:06d}.pth'.format(args.SAVE_ROOT, epoch))
    torch.save(optimizer.state_dict(), '{:s}/optimizer_{:06d}.pth'.format(args.SAVE_ROOT, epoch))
    # if epoch % 4 == 0:  
    #     logger.info('Saving state, epoch:' + str(epoch))
    #     torch.save(net.state_dict(), '{:s}/model_{:06d}.pth'.format(args.SAVE_ROOT, epoch))
    #     torch.save(optimizer.state_dict(), '{:s}/optimizer_{:06d}.pth'.format(args.SAVE_ROOT, epoch))

    return iteration


def run_val(args, val_data_loader, val_dataset, net, epoch, iteration):
        torch.cuda.synchronize()
        tvs = time.perf_counter()

        mAP, ap_all, ap_strs = validate(args, net, val_data_loader, val_dataset, epoch)
        label_types = args.label_types #+ ['ego_action']
        all_classes = args.all_classes #+ [args.ego_classes]
        mAP_group = dict()

        # for nlt in range(args.num_label_types+1):
        for nlt in range(args.num_label_types):
            for ap_str in ap_strs[nlt]:
                logger.info(ap_str)
            ptr_str = '\n{:s} MEANAP:::=> {:0.5f}'.format(label_types[nlt], mAP[nlt])
            logger.info(ptr_str)

            if args.TENSORBOARD:
                mAP_group[label_types[nlt]] = mAP[nlt]
                # args.sw.add_scalar('{:s}mAP'.format(label_types[nlt]), mAP[nlt], iteration)
                class_AP_group = dict()
                for c, ap in enumerate(ap_all[nlt]):
                    class_AP_group[all_classes[nlt][c]] = ap
                args.sw.add_scalars('ClassAP-{:s}'.format(label_types[nlt]), class_AP_group, epoch)

        if args.TENSORBOARD:
            args.sw.add_scalars('mAPs', mAP_group, epoch)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        prt_str = '\nValidation TIME::: {:0.3f}\n\n'.format(t0-tvs)
        logger.info(prt_str)




