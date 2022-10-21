import os
import time
import random
from pathlib import Path

import numpy as np
import logging
import argparse
import shutil

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.optim.lr_scheduler as lr_scheduler

import wandb

from util import config
from util.s3dis import S3DIS
from util.common_util import AverageMeter, intersectionAndUnionGPU, find_free_port
from util.data_util import collate_fn
from util import transform as t


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Point Cloud Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/s3dis/s3dis_pointtransformer_repro.yaml',
                        help='config file')
    parser.add_argument('opts', help='see config/s3dis/s3dis_pointtransformer_repro.yaml for all options', default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)


def main_process():
    return not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


def define_wandb_metrics():
    wandb.define_metric('loss_train', summary='min')
    wandb.define_metric('mIoU_train', summary='max')
    wandb.define_metric('mAcc_train', summary='max')
    wandb.define_metric('allAcc_train', summary='max')

    wandb.define_metric('loss_val', summary='min')
    wandb.define_metric('mIoU_val', summary='max')
    wandb.define_metric('mAcc_val', summary='max')
    wandb.define_metric('allAcc_val', summary='max')

    wandb.define_metric('loss_train_batch', summary='min')
    wandb.define_metric('mIoU_train_batch', summary='max')
    wandb.define_metric('mAcc_train_batch', summary='max')
    wandb.define_metric('allAcc_train_batch', summary='max')


def main():
    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    # Initialise wandb
    os.environ["WANDB_MODE"] = "dryrun"
    wandb.init(project="point-transformer", name=str(Path(__file__).parts[-3:-1]))
    if len(wandb.config.__dict__) != 0:
        config = wandb.config
        args.update(config)
    args.update({"batch_size": 150000//args.voxel_max})
    wandb.config.update(args)
    wandb.config.update({"pretrained": "freeze_body" in args.keys()})
    define_wandb_metrics()

    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False

    if args.data_name == 's3dis':
        S3DIS(split='train', data_root=args.data_root, test_area=args.test_area)
        S3DIS(split='val', data_root=args.data_root, test_area=args.test_area)
    elif args.data_name == 'church':
        S3DIS(split='train', data_root=args.data_root, test_area=args.test_area)
        S3DIS(split='val', data_root=args.data_root, test_area=args.test_area)
    else:
        print(f"Err dataset {args.data_name} not yet implemented")
        raise NotImplementedError()
    if args.multiprocessing_distributed:
        port = find_free_port()
        args.dist_url = f"tcp://localhost:{port}"
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
    global args, best_iou
    args, best_iou = argss, 0
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size,
                                rank=args.rank)

    if args.arch == 'pointtransformer_seg_repro':
        from model.pointtransformer.pointtransformer_seg import pointtransformer_seg_repro as Model
    else:
        raise Exception('architecture not supported yet'.format(args.arch))
    model = Model(c=args.fea_dim, k=args.classes)
    if args.sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    if hasattr(args, "optimizer"):
        if args.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.epochs * 0.6), int(args.epochs * 0.8)],
                                         gamma=0.1)
    # If we're using warmup , could also use a sequentialLR see:
    # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.SequentialLR.html#torch.optim.lr_scheduler.SequentialLR
    if hasattr(args, "scheduler"):
        if args.scheduler == "warmup":
            # From Marc Katzenmaier at UZH
            def lambda_poly_lr_schedule_warmup(max_epoch, power, warmup_length):
                # lr schedule WarmupPlolyLR similar to deeplab
                # Reference: https://github.com/tensorflow/models/blob/21b73d22f3ed05b650e85ac50849408dd36de32e/research/deeplab/utils/train_utils.py#L337  # noqa
                import math
                def schedule(epoch):
                    if epoch < warmup_length:
                        return math.pow((1.0 - epoch / max_epoch), power) * (
                                (epoch + 1) / float(warmup_length + 1))  # add the plus 1 so it won't start with 0
                    else:
                        return math.pow((1.0 - epoch / max_epoch), power)

                return schedule

            power = 0.9 if not hasattr(args, "power") else args.power
            warmup_length = 10 if not hasattr(args, "warmup_length") else args.warmup_length
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                          lr_lambda=lambda_poly_lr_schedule_warmup(args.epochs,
                                                                                                   power=0.9,
                                                                                                   warmup_length=10))
            print("Using warmup from Marc")

    if main_process():
        global logger
        logger = get_logger()
        logger.info(args)
        logger.info("=> creating model ...")
        logger.info("Classes: {}".format(args.classes))
        logger.info(model)
    if args.distributed:
        torch.cuda.set_device(gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(
            model.cuda(),
            device_ids=[gpu],
            find_unused_parameters=True if "transformer" in args.arch else False
        )

    else:
        model = torch.nn.DataParallel(model.cuda())

    if args.weight:
        if os.path.isfile(args.weight):
            if main_process():
                logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight)
            model.load_state_dict(checkpoint['state_dict'])
            if main_process():
                logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            logger.info("=> no weight found at '{}'".format(args.weight))

    if args.resume:
        if os.path.isfile(args.resume):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            # best_iou = 40.0
            best_iou = checkpoint['best_iou']
            if main_process():
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            if main_process():
                logger.info("=> no checkpoint found at '{}'".format(args.resume))

    if hasattr(args, "freeze_body") and hasattr(args, "weight") and args.freeze_body and args.weight:
        logger.info("Freezing body of model")
        for p in model.parameters():
            p.requires_grad = False
        for p in model.module.cls.parameters():
            p.requires_grad = True

    if args.data_name == "s3dis":
        train_transform = t.Compose(
            [t.RandomScale([0.9, 1.1]), t.ChromaticAutoContrast(), t.ChromaticTranslation(), t.ChromaticJitter(),
             t.HueSaturationTranslation()])
    else:
        train_transform = t.Compose([t.RandomScale([0.9, 1, 1])])
    train_data = S3DIS(split='train', data_root=args.data_root, test_area=args.test_area, voxel_size=args.voxel_size,
                       voxel_max=args.voxel_max, transform=train_transform, shuffle_index=True, loop=args.loop,
                       dupe_intensity=args.fea_dim == 6 and args.data_name == "church")
    if main_process():
        logger.info("train_data samples: '{}'".format(len(train_data)))
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=(train_sampler is None),
                                               num_workers=args.workers, pin_memory=True, sampler=train_sampler,
                                               drop_last=True, collate_fn=collate_fn)

    val_loader = None
    if args.evaluate:
        val_transform = None
        val_data = S3DIS(split='val', data_root=args.data_root, test_area=args.test_area, voxel_size=args.voxel_size,
                         voxel_max=800000, transform=val_transform,
                         dupe_intensity=args.fea_dim == 6 and args.data_name == "church")
        if args.distributed:
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
        else:
            val_sampler = None
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False,
                                                 num_workers=args.workers, pin_memory=True, sampler=val_sampler,
                                                 collate_fn=collate_fn)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        loss_train, mIoU_train, mAcc_train, allAcc_train = train(train_loader, model, criterion, optimizer, epoch)
        scheduler.step()
        wandb.log({'lr': scheduler.get_last_lr()[-1]}, commit=False)
        epoch_log = epoch + 1
        if main_process():
            wandb.log({'loss_train': loss_train,
                       'mIoU_train': mIoU_train,
                       'mAcc_train': mAcc_train,
                       'allAcc_train': allAcc_train,
                       "epoch_log": epoch_log}, commit=False)

        is_best = False
        if args.evaluate and (epoch_log % args.eval_freq == 0):
            if args.data_name == 'shapenet':
                raise NotImplementedError()
            else:
                loss_val, mIoU_val, mAcc_val, allAcc_val = validate(val_loader, model, criterion)

            if main_process():
                wandb.log({'loss_val': loss_val,
                           'mIoU_val': mIoU_val,
                           'mAcc_val': mAcc_val,
                           'allAcc_val': allAcc_val,
                           "epoch_log": epoch_log})
                is_best = mIoU_val > best_iou
                best_iou = max(best_iou, mIoU_val)

        if (epoch_log % args.save_freq == 0) and main_process():
            filename = args.save_path + '/model/model_last.pth'
            logger.info('Saving checkpoint to: ' + filename)
            torch.save({'epoch': epoch_log, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(), 'best_iou': best_iou, 'is_best': is_best}, filename)
            if is_best:
                logger.info('Best validation mIoU updated to: {:.4f}'.format(best_iou))
                shutil.copyfile(filename, args.save_path + '/model/model_best.pth')

    if main_process():
        logger.info('==>Training done!\nBest Iou: %.3f' % (best_iou))


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)
    for i, (coord, feat, target, offset) in enumerate(train_loader):  # (n, 3), (n, c), (n), (b)
        data_time.update(time.time() - end)
        coord, feat, target, offset = coord.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(
            non_blocking=True), offset.cuda(non_blocking=True)
        # Batches are concatenated together (no batch dim) and offset is the indices of the batchs
        output = model([coord, feat, offset])
        if target.shape[-1] == 1:
            target = target[:, 0]  # for cls
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.max(1)[1]
        n = coord.size(0)
        if args.multiprocessing_distributed:
            loss *= n
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            loss /= n
        if hasattr(args, "save_train_output"):
            torch.save((coord.cpu(), feat.cpu(), target.cpu(), offset.cpu(), output.cpu()), f"E{epoch+1}I{i+1}.pt")
        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        if args.multiprocessing_distributed:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()

        # calculate remain time
        current_iter = epoch * len(train_loader) + i + 1
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'Loss {loss_meter.val:.4f} '
                        'Accuracy {accuracy:.4f}.'.format(epoch + 1, args.epochs, i + 1, len(train_loader),
                                                          batch_time=batch_time, data_time=data_time,
                                                          remain_time=remain_time,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))
        if main_process():
            wandb.log({'loss_train_batch': loss_meter.val,
                       'mIoU_train_batch': np.mean(intersection / (union + 1e-10)),
                       'mAcc_train_batch': np.mean(intersection / (target + 1e-10)),
                       'allAcc_train_batch': accuracy,
                       "current_iter": current_iter}, commit=False)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if main_process():
        logger.info(
            'Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch + 1, args.epochs, mIoU,
                                                                                           mAcc, allAcc))
    return loss_meter.avg, mIoU, mAcc, allAcc


def validate(val_loader, model, criterion):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.eval()
    end = time.time()
    for i, (coord, feat, target, offset) in enumerate(val_loader):
        data_time.update(time.time() - end)
        coord, feat, target, offset = coord.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(
            non_blocking=True), offset.cuda(non_blocking=True)
        if target.shape[-1] == 1:
            target = target[:, 0]  # for cls
        with torch.no_grad():
            output = model([coord, feat, offset])
        loss = criterion(output, target)

        output = output.max(1)[1]
        n = coord.size(0)
        if args.multiprocessing_distributed:
            loss *= n
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            loss /= n

        if hasattr(args, "save_val_output"):
            torch.save((coord.cpu(), feat.cpu(), target.cpu(), offset.cpu(), output.cpu()), f"VI{i+1}.pt")

        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        if args.multiprocessing_distributed:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        'Accuracy {accuracy:.4f}.'.format(i + 1, len(val_loader),
                                                          data_time=data_time,
                                                          batch_time=batch_time,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    if main_process():
        logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
        for i in range(args.classes):
            logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

    return loss_meter.avg, mIoU, mAcc, allAcc


if __name__ == '__main__':
    import gc

    gc.collect()
    main()
