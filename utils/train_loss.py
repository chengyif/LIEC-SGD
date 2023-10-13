import torch
import time
from .meters import AverageMeter, ProgressMeter
from .accuracy import Accuracy

def Train_loss(train_loader, model, criterion, args, epoch, prefix=''):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':6.3f')
    #top1 = AverageMeter('Acc@1', ':6.2f')
    #top5 = AverageMeter('Acc@5', ':6.2f')
    #progress = ProgressMeter(
    #    len(val_loader),
    #    [batch_time, losses, top1, top5],
    #    prefix="Epoch: [{}] Test: ".format(epoch))
    
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses],
        prefix="Epoch: [{}] Train: ".format(epoch))

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (images, target) in enumerate(train_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            #acc1, acc5 = Accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            #top1.update(acc1[0], images.size(0))
            #top5.update(acc5[0], images.size(0))

            # measure elapsed time
            #epoch_time.update((time.time() - end)*len(val_loader))
            #end = time.time()

        progress.display(i, prefix=prefix)

    return losses.avg
