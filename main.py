import argparse
import os
import sys
import random
import shutil
import time
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import pandas as pd
from utils import *
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors, \
    _take_tensors
import compressor

record = []

def main():
    args = GetArgs()
    print(args, flush=True)

    if args.seed is not None:
        setup_seed(args.seed)
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    args.world_size = args.process_num * args.gpus * args.world_size
    print('world_size:',args.world_size)    
    mp.spawn(main_worker, nprocs=args.gpus * args.process_num, args=(args,))

def main_worker(process, args):
    setup_seed(args.seed)
    args.rank = args.rank * args.gpus * args.process_num + process
    gpu = process//args.process_num + args.st
    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    dist.init_process_group(backend=args.dist_backend,
                            init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.dataset == 'cifar10':
        import models
        model = models.__dict__[args.arch](num_classes=10)
    elif args.dataset == 'cifar100':
        import models
        model = models.__dict__[args.arch](num_classes=100)
    elif args.dataset == 'tiny-imagenet':
        import models
        model = models.__dict__[args.arch](num_classes=200)
    elif args.dataset == 'imagenet':
        import torchvision.models as models
        model = models.__dict__[args.arch]()
    else:
        assert False

    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)
    
    # sync the model among workers
    for k, v in model.state_dict().items():
        dist.broadcast(v.data, src=args.root)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optimizer == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), args.lr,
                                        weight_decay=args.weight_decay)
    else:
        assert False

    # Data loading code
    train_dataset, val_dataset = GetDataset(args.dataset, args.path)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True,
        seed=args.seed)
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True,
        seed=args.seed)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=test_sampler)

    # residuals for dist-r
    old_p = [p.data.clone().detach().zero_() for p in model.parameters()]
    all_p = []
    residuals = []
    s_residuals = []
    residuals_sum = []
    bucket_size = args.bucket_size

    for p in model.parameters():
        all_p.append(p.data)
    dev_data_buckets = _take_tensors(all_p, bucket_size)
    for i, dev_data in enumerate(dev_data_buckets):
        p_new = _flatten_dense_tensors(dev_data)
        residuals.append(torch.zeros_like(p_new))
        s_residuals.append(torch.zeros_like(p_new))
        residuals_sum.append(torch.zeros_like(p_new))
    train_time = 0
    average_period = args.average_period
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        lr = adjust_learning_rate(optimizer, epoch, args)
        if process==0:
            print('Current lr: {:.6f}'.format(lr), flush=True)

        # train for one epoch
        train_loss, epoch_time, error_norm = train(train_loader, model, criterion, optimizer, epoch, args,
              old_p, residuals, s_residuals, residuals_sum, average_period, bucket_size)
        train_time += epoch_time
        if process == 0:
            print ("Training time:", train_time)

        # evaluate on validation set
        acc1, test_loss = Validate(test_loader, model, criterion, args, epoch, 
                            prefix='[worker '+str(args.rank)+']')
        dist.reduce(acc1, dst=args.root, op=dist.ReduceOp.SUM)
        if args.rank == args.root:
            acc1 /= args.world_size
        test_loss = torch.tensor(test_loss)
        test_loss = test_loss.cuda(args.gpu)
        dist.reduce(test_loss, dst=args.root, op=dist.ReduceOp.SUM)
        if args.rank == args.root:
            test_loss /= args.world_size

        if process == 0:
            print('Test accuracy: {:6.2f}'.format(acc1), flush=True)
            record.append([epoch+1, train_loss, acc1.item(), test_loss.item(), error_norm, train_time])

    
    if process == 0:
        record_csv = pd.DataFrame(record)
        record_header = ['Epoch', 'Train Loss', 'Test Accuracy', 'Test Loss', 'Error Norm', 'Train Time']
        record_csv.to_csv("./result/" + args.dataset + "/method_" + str(args.method) + "_model_" + str(args.arch) + "_bucket_size_" + str(bucket_size) + "_average_period_" + str(average_period) + "_seed_" + str(args.seed) + "_record.csv", index=False, header=record_header)


def train(train_loader, model, criterion, optimizer, epoch, args,
          old_p, residuals, s_residuals, residuals_sum, average_period, bucket_size):
    losses = AverageMeter('Loss', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    p_names = [name for name, p in model.named_parameters()]

    if epoch > 0 and (args.method.startswith('liec') or args.method.startswith('cser')):
        for idx, p in enumerate(model.parameters()):
            p.data.copy_(old_p[idx])

    neolithic_r = 2
    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        cur_iter = i + epoch * len(train_loader)        
        images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        optimizer.zero_grad()

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = Accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        loss.backward()
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if not k in p_names: # broadcast non-param buffers from rank 0
                    dist.broadcast(v.data, src=args.root)

            if args.method == 'psgd':
                # reduce
                for p in model.parameters():
                    dist.reduce(p.grad, dst=args.root, op=dist.ReduceOp.SUM)
                for p in model.parameters():    
                    if args.rank == args.root:
                        p.grad /= args.world_size
                # broadcast
                for p in model.parameters():
                    dist.broadcast(p.grad, src=args.root)
                optimizer.step()


            elif args.method == 'doublesqueeze':
                all_grads = []
                for p in model.parameters():
                    all_grads.append(p.grad.data)
                dev_grads_buckets = _take_tensors(all_grads, bucket_size)
                
                for i, dev_grads in enumerate(dev_grads_buckets):
                    d_p_new = _flatten_dense_tensors(dev_grads)
                    err_buf = residuals[i]
                    server_err_buf = s_residuals[i]
                    d_p_new.add_(err_buf)
                    p_buf = d_p_new
                    d_p_new_scale = torch.ones(1)
                    d_p_new_scale[0] = d_p_new.abs().sum().cpu().item()/d_p_new.numel()
                    operator = compressor.compressor(using_cuda = True, local_rank = args.rank, device_num = args.gpu)
                    d_p_new, tensor_size = operator.compress(d_p_new)
                    tmp = operator.uncompress(d_p_new.clone(), tensor_size)
                    tmp.mul_(d_p_new_scale.item())
                    err_buf.copy_(p_buf).sub_(tmp)

                    if args.rank == args.root:
                        d_p_new_list = [torch.zeros_like(d_p_new) for _ in range(args.world_size)]
                        d_p_new_scale_list = [torch.zeros_like(d_p_new_scale) for _ in range(args.world_size)]
                    if args.rank == args.root:
                        dist.gather(d_p_new, d_p_new_list)
                    else: 
                        dist.gather(d_p_new)
                    if args.rank == args.root:
                        dist.gather(d_p_new_scale, d_p_new_scale_list)
                    else: 
                        dist.gather(d_p_new_scale)

                    if args.rank == args.root:
                        d_p_new = torch.zeros(tensor_size).cuda()
                        for d_p, d_p_scale in zip(d_p_new_list, d_p_new_scale_list):
                            tmp = operator.uncompress(d_p, tensor_size)
                            d_p_new.add_(tmp, alpha=d_p_scale.item())
                        d_p_new /= args.world_size
                        d_p_new.add_(server_err_buf)
                        un_compr = d_p_new
                        d_p_new_scale = torch.ones(1)
                        d_p_new_scale[0] = d_p_new.abs().sum().cpu().item()/d_p_new.numel()
                        d_p_new, _ = operator.compress(d_p_new)
                        tmp = operator.uncompress(d_p_new.clone(), tensor_size)
                        tmp.mul_(d_p_new_scale.item())
                        server_err_buf.copy_(un_compr).sub_(tmp)
  
                    dist.broadcast(d_p_new, src=args.root)
                    dist.broadcast(d_p_new_scale, src=args.root)
                    d_p_new = operator.uncompress(d_p_new, tensor_size)
                    d_p_new.mul_(d_p_new_scale.item())

                    dev_grads_new = _unflatten_dense_tensors(d_p_new,dev_grads)
                    for grad, reduced in zip(dev_grads, dev_grads_new):
                        grad.copy_(reduced)
                optimizer.step()

            elif args.method == 'memsgd':
                all_grads = []
                for p in model.parameters():
                    all_grads.append(p.grad.data)
                dev_grads_buckets = _take_tensors(all_grads, bucket_size)
                
                for i, dev_grads in enumerate(dev_grads_buckets):
                    d_p_new = _flatten_dense_tensors(dev_grads)
                    err_buf = residuals[i]
                    d_p_new.add_(err_buf)
                    p_buf = d_p_new
                    d_p_new_scale = torch.ones(1)
                    d_p_new_scale[0] = d_p_new.abs().sum().cpu().item()/d_p_new.numel()
                    operator = compressor.compressor(using_cuda = True, local_rank = args.rank, device_num = args.gpu)
                    d_p_new, tensor_size = operator.compress(d_p_new)
                    tmp = operator.uncompress(d_p_new.clone(), tensor_size)
                    tmp.mul_(d_p_new_scale.item())
                    err_buf.copy_(p_buf).sub_(tmp)

                    if args.rank == args.root:
                        d_p_new_list = [torch.zeros_like(d_p_new) for _ in range(args.world_size)]
                        d_p_new_scale_list = [torch.zeros_like(d_p_new_scale) for _ in range(args.world_size)]
                    if args.rank == args.root:
                        dist.gather(d_p_new, d_p_new_list)
                    else: 
                        dist.gather(d_p_new)
                    if args.rank == args.root:
                        dist.gather(d_p_new_scale, d_p_new_scale_list)
                    else: 
                        dist.gather(d_p_new_scale)

                    d_p_new = torch.zeros(tensor_size).cuda()
                    if args.rank == args.root:                        
                        for d_p, d_p_scale in zip(d_p_new_list, d_p_new_scale_list):
                            tmp = operator.uncompress(d_p, tensor_size)
                            d_p_new.add_(tmp, alpha=d_p_scale.item())
                        d_p_new /= args.world_size
                    dist.broadcast(d_p_new, src=args.root)
                    dev_grads_new = _unflatten_dense_tensors(d_p_new,dev_grads)
                    for grad, reduced in zip(dev_grads, dev_grads_new):
                        grad.copy_(reduced)
                optimizer.step()


            elif args.method == 'liec':
                all_grads = []
                for p in model.parameters():
                    all_grads.append(p.grad.data)
                dev_grads_buckets = _take_tensors(all_grads, bucket_size)

                if (cur_iter+1) % average_period == 0:
                    for idx, p in enumerate(model.parameters()):
                        dist.reduce(p.data, dst=args.root, op=dist.ReduceOp.SUM)
                        if args.rank == args.root:
                            p.data /= args.world_size
                        dist.broadcast(p.data, src=args.root)
                    for i, dev_grads in enumerate(dev_grads_buckets):
                        d_p_new = _flatten_dense_tensors(dev_grads)
                        dist.reduce(d_p_new, dst=args.root, op=dist.ReduceOp.SUM)
                        if args.rank == args.root:
                            d_p_new /= args.world_size
                            d_p_new += s_residuals[i]
                            s_residuals[i].zero_()
                        dist.broadcast(d_p_new, src=args.root)
                        dev_grads_new = _unflatten_dense_tensors(d_p_new,dev_grads)
                        for grad, reduced in zip(dev_grads, dev_grads_new):
                            grad.copy_(reduced)
                else:
                    for i, dev_grads in enumerate(dev_grads_buckets):
                        d_p_new = _flatten_dense_tensors(dev_grads)
                        p_buf = d_p_new
                        server_err_buf = s_residuals[i]
                        d_p_new_scale = torch.ones(1)
                        d_p_new_scale[0] = d_p_new.abs().sum().cpu().item()/d_p_new.numel()
                        operator = compressor.compressor(using_cuda = True, local_rank = args.rank, device_num = args.gpu)
                        d_p_new, tensor_size = operator.compress(d_p_new)
                        tmp = operator.uncompress(d_p_new.clone(), tensor_size)
                        tmp.mul_(d_p_new_scale.item())
                        residuals[i] = p_buf.sub_(tmp)

                        if args.rank == args.root:
                            d_p_new_list = [torch.zeros_like(d_p_new) for _ in range(args.world_size)]
                            d_p_new_scale_list = [torch.zeros_like(d_p_new_scale) for _ in range(args.world_size)]
                        if args.rank == args.root:
                            dist.gather(d_p_new, d_p_new_list)
                        else: 
                            dist.gather(d_p_new)
                        if args.rank == args.root:
                            dist.gather(d_p_new_scale, d_p_new_scale_list)
                        else: 
                            dist.gather(d_p_new_scale)

                        if args.rank == args.root:
                            d_p_new = torch.zeros(tensor_size).cuda()
                            for d_p, d_p_scale in zip(d_p_new_list, d_p_new_scale_list):
                                tmp = operator.uncompress(d_p, tensor_size)
                                d_p_new.add_(tmp, alpha=d_p_scale.item())
                            d_p_new /= args.world_size
                            d_p_new.add_(server_err_buf)
                            un_compr = d_p_new
                            d_p_new_scale = torch.ones(1)
                            d_p_new_scale[0] = d_p_new.abs().sum().cpu().item()/d_p_new.numel()
                            d_p_new, _ = operator.compress(d_p_new)
                            tmp = operator.uncompress(d_p_new.clone(), tensor_size)
                            tmp.mul_(d_p_new_scale.item())
                            server_err_buf.copy_(un_compr).sub_(tmp)

                        dist.broadcast(d_p_new, src=args.root)
                        dist.broadcast(d_p_new_scale, src=args.root)
                        d_p_new = operator.uncompress(d_p_new, tensor_size)
                        d_p_new.mul_(d_p_new_scale.item())
                        d_p_new.add_(residuals[i])
                        dev_grads_new = _unflatten_dense_tensors(d_p_new,dev_grads)
                        for grad, reduced in zip(dev_grads, dev_grads_new):
                            grad.copy_(reduced)
                optimizer.step() 
                  
            elif args.method == 'cser':
                all_grads = []
                for p in model.parameters():
                    all_grads.append(p.grad.data)
                dev_grads_buckets = _take_tensors(all_grads, bucket_size)
                
                for i, dev_grads in enumerate(dev_grads_buckets):
                    d_p_new = _flatten_dense_tensors(dev_grads)
                    p_buf = d_p_new
                    resi_sum_buf = residuals_sum[i]
                    d_p_new_scale = torch.ones(1)
                    d_p_new_scale[0] = d_p_new.abs().sum().cpu().item()/d_p_new.numel()
                    operator = compressor.compressor(using_cuda = True, local_rank = args.rank, device_num = args.gpu)
                    d_p_new, tensor_size = operator.compress(d_p_new)
                    tmp = operator.uncompress(d_p_new.clone(), tensor_size)
                    tmp.mul_(d_p_new_scale.item())
                    residuals[i] = p_buf.sub_(tmp) #r_{i,t}
                    resi_sum_buf.copy_(resi_sum_buf.sub_(residuals[i]))

                    if args.rank == args.root:
                        d_p_new_list = [torch.zeros_like(d_p_new) for _ in range(args.world_size)]
                        d_p_new_scale_list = [torch.zeros_like(d_p_new_scale) for _ in range(args.world_size)]
                    if args.rank == args.root:
                        dist.gather(d_p_new, d_p_new_list)
                    else: 
                        dist.gather(d_p_new)
                    if args.rank == args.root:
                        dist.gather(d_p_new_scale, d_p_new_scale_list)
                    else: 
                        dist.gather(d_p_new_scale)

                    d_p_new = torch.zeros(tensor_size).cuda()
                    if args.rank == args.root:                        
                        for d_p, d_p_scale in zip(d_p_new_list, d_p_new_scale_list):
                            tmp = operator.uncompress(d_p, tensor_size)
                            d_p_new.add_(tmp, alpha=d_p_scale.item())
                        d_p_new /= args.world_size
                    dist.broadcast(d_p_new, src=args.root)
                    d_p_new.add_(residuals[i])
                    if (cur_iter+1) % average_period == 0:
                        resi_new = residuals_sum[i]
                        resi_buf = resi_new
                        d_p_new.add_(resi_buf)
                        resi_compre_scale = torch.ones(1)
                        resi_compre_scale[0] = resi_buf.abs().sum().cpu().item()/resi_buf.numel()
                        resi_compre, tensor_size = operator.compress(resi_buf)
                        resi_tmp = operator.uncompress(resi_compre.clone(), tensor_size)
                        resi_tmp.mul_(resi_compre_scale.item())
                        resi_new.copy_(resi_buf.sub_(resi_tmp))

                        if args.rank == args.root:
                            resi_compre_list = [torch.zeros_like(resi_compre) for _ in range(args.world_size)]
                            resi_compre_scale_list = [torch.zeros_like(resi_compre_scale) for _ in range(args.world_size)]
                        if args.rank == args.root:
                            dist.gather(resi_compre, resi_compre_list)
                        else: 
                            dist.gather(resi_compre)
                        if args.rank == args.root:
                            dist.gather(resi_compre_scale, resi_compre_scale_list)
                        else: 
                            dist.gather(resi_compre_scale)

                        resi_buf_new = torch.zeros(tensor_size).cuda()
                        if args.rank == args.root:                        
                            for resi_compre, resi_compre_scale in zip(resi_compre_list, resi_compre_scale_list):
                                tmp = operator.uncompress(resi_compre, tensor_size)
                                resi_buf_new.add_(tmp, alpha=resi_compre_scale.item())
                            resi_buf_new /= args.world_size
                        dist.broadcast(resi_buf_new, src=args.root)
                        resi_buf_new.add_(resi_new)
                        d_p_new.sub_(resi_buf_new)


                    dev_grads_new = _unflatten_dense_tensors(d_p_new,dev_grads)
                    for grad, reduced in zip(dev_grads, dev_grads_new):
                        grad.copy_(reduced)
                optimizer.step()

            elif args.method == 'neolithic':
                all_grads = []
                for p in model.parameters():
                    all_grads.append(p.grad.data)
                dev_grads_buckets = _take_tensors(all_grads, bucket_size)

                for i, dev_grads in enumerate(dev_grads_buckets):
                    d_p_new = _flatten_dense_tensors(dev_grads)
                    compr_sum = torch.zeros_like(d_p_new)
                    err_buf = residuals[i]
                    server_err_buf = s_residuals[i]
                    d_p_new.add_(err_buf)
                    p_buf = d_p_new
                    d_p_new_list = []
                    d_p_new_scale_list = []
                    if args.rank == args.root:
                        compr_diff_list_list = []
                        diff_scale_list_list = []
                    for _ in range(neolithic_r):
                        diff = d_p_new - compr_sum
                        diff_scale = torch.ones(1)
                        diff_scale[0] = diff.abs().sum().cpu().item()/diff.numel()
                        operator = compressor.compressor(using_cuda = True, local_rank = args.rank, device_num = args.gpu)
                        compr_diff, tensor_size = operator.compress(diff)
                        d_p_new_list.append(compr_diff)
                        d_p_new_scale_list.append(diff_scale)
                        if args.rank == args.root:
                            compr_diff_list = [torch.zeros_like(compr_diff) for _ in range(args.world_size)]
                            diff_scale_list = [torch.zeros_like(diff_scale) for _ in range(args.world_size)]
                        if args.rank == args.root:
                            dist.gather(compr_diff, compr_diff_list)
                        else: 
                            dist.gather(compr_diff)
                        if args.rank == args.root:
                            dist.gather(diff_scale, diff_scale_list)
                        else: 
                            dist.gather(diff_scale)
                        if args.rank == args.root:
                            compr_diff_list_list.append(compr_diff_list)
                            diff_scale_list_list.append(diff_scale_list)

                        tmp = operator.uncompress(compr_diff.clone(), tensor_size)
                        tmp.mul_(diff_scale.item())
                        compr_sum.add_(tmp)
                    err_buf.copy_(p_buf).sub_(compr_sum)
                    if args.rank == args.root:
                        d_p_new = torch.zeros(tensor_size).cuda()
                        for j in range(neolithic_r):
                            for d_p, d_p_scale in zip(compr_diff_list_list[j], diff_scale_list_list[j]):
                                tmp = operator.uncompress(d_p, tensor_size)
                                d_p_new.add_(tmp, alpha=d_p_scale.item())
                        d_p_new /= args.world_size
                        d_p_new.add_(server_err_buf)
                        server_compr_sum = torch.zeros_like(d_p_new)
                        un_compr = d_p_new
                    for j in range(neolithic_r):
                        if args.rank == args.root:
                            diff = d_p_new - server_compr_sum
                            diff_scale = torch.ones(1)
                            diff_scale[0] = diff.abs().sum().cpu().item()/diff.numel()
                            compr_diff, tensor_size = operator.compress(diff)
                            tmp = operator.uncompress(compr_diff.clone(), tensor_size)
                            tmp.mul_(diff_scale.item())
                            server_compr_sum.add_(tmp)
                            d_p_new_list[j].copy_(compr_diff)
                            d_p_new_scale_list[j].copy_(diff_scale)
                        dist.broadcast(d_p_new_list[j], src=args.root)                            
                        dist.broadcast(d_p_new_scale_list[j], src=args.root)
                    if args.rank == args.root:
                        server_err_buf.copy_(un_compr).sub_(server_compr_sum)
                    d_p_new.zero_()                 
                    for j in range(neolithic_r):
                        uncompr_d_p_new = operator.uncompress(d_p_new_list[j], tensor_size)
                        uncompr_d_p_new.mul_(d_p_new_scale_list[j].item())
                        d_p_new.add_(uncompr_d_p_new)

                    dev_grads_new = _unflatten_dense_tensors(d_p_new,dev_grads)
                    for grad, reduced in zip(dev_grads, dev_grads_new):
                        grad.copy_(reduced)
                optimizer.step()
 
            else:
                assert False

    train_time = time.time() - end

    progress.display(i, prefix='[worker '+str(args.rank)+']')
    losses.avg = torch.from_numpy(np.array(losses.avg))
    losses.avg = losses.avg.cuda(args.gpu)
    dist.reduce(losses.avg, dst=args.root, op=dist.ReduceOp.SUM)
    if args.rank == args.root:
        losses.avg /= args.world_size
    
    train_time = torch.tensor(train_time)
    train_time = train_time.cuda(args.gpu)
    dist.reduce(train_time, dst=args.root, op=dist.ReduceOp.MAX) #choose the maximal value of the time of one epoch across all processes

    error_norm = 0
    if args.method == 'psgd':
        error_norm = 0   
    elif args.method == 'memsgd':
        residuals_cat = torch.cat(residuals)
        dist.reduce(residuals_cat, dst=args.root, op=dist.ReduceOp.SUM)
        if args.rank == args.root:
            residuals_cat /= args.world_size
            error_norm = residuals_cat.norm().item()                    
    elif args.method == 'doublesqueeze' or args.method == 'neolithic':
        residuals_cat = torch.cat(residuals)
        dist.reduce(residuals_cat, dst=args.root, op=dist.ReduceOp.SUM)
        if args.rank == args.root:
            residuals_cat /= args.world_size
            residuals_cat += torch.cat(s_residuals)
            error_norm = residuals_cat.norm().item() 

    elif args.method.startswith('liec'):
        if args.rank == args.root:
            error_norm = torch.cat(s_residuals).norm().item()

    elif args.method.startswith('cser'):
        residuals_cat = torch.cat(residuals_sum)
        dist.reduce(residuals_cat, dst=args.root, op=dist.ReduceOp.SUM)
        if args.rank == args.root:
            residuals_cat /= args.world_size
            error_norm = residuals_cat.norm().item()

    if args.method.startswith('liec') or args.method.startswith('cser'): #use the average model across the worker to do testing
        for idx, p in enumerate(model.parameters()):
            old_p[idx].copy_(p.data)
        for idx, p in enumerate(model.parameters()):
            dist.reduce(p.data, dst=args.root, op=dist.ReduceOp.SUM)
            if args.rank == args.root:
                p.data /= args.world_size
            dist.broadcast(p.data, src=args.root)

    return losses.avg.item(), train_time.item(), error_norm


def adjust_learning_rate(optimizer, epoch, args):
    assert len(args.decay_schedule) >= 2

    if epoch < args.decay_schedule[0]:
        lr = args.lr
    elif epoch < args.decay_schedule[1]:
        lr = args.lr * 0.1
    else:
        lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.deterministic = True


if __name__ == '__main__':
    main()
