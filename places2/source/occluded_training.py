# this code is modified from the pytorch example code on https://github.com/CSAILVision/places365

import argparse
import os
import shutil
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Places2 Training Using Occluded Images')
parser.add_argument('--data', default='places2/images/complete_set', help='path to places2 dataset')
parser.add_argument('-j', '--workers', default=6, type=int, metavar='N',
                    help='number of data loading workers (default: 6)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--occlusion_method', default='cutout', type=str,
                    help="Which occlusion method to use. Should be either 'cutout', 'irreg', 'hso', or 'randerase'")
parser.add_argument('--num_cutout_masks', default=6, type=int,
                    help="Number of cutout masks to use per image when using occlusion method 'cutout'")
parser.add_argument('--cutout_mask_length', default=56, type=int,
                    help="Length and width of each mask when using occlusion method 'cutout'")
parser.add_argument('--irreg_percent_masked', default=.4, type=float,
                    help="portion of image that is masked when using occlusion method 'irreg'")
parser.add_argument('--hso_mask_dir', default='human_masks', type=str,
                    help="location of the human-shaped masks when using occlusion method 'hso'")
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--gpu', type=int, default=[0], nargs ='+', help ='used gpu')

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()
    print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in args.gpu)
    # create model
    print("=> creating model '{}'".format('resnet18'))
    model = models.__dict__["resnet18"](num_classes=365)
    model = torch.nn.DataParallel(model).cuda()
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    human_masks = []
    means = []

    # Apply appropriate image transformations based on occluded training type
    if args.train_method != 'hso':
        train_transforms = [transforms.RandomSizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor()]
        if args.train_method == 'randerase':
            print('Adding Random Erasing Transform')
            train_transforms.append(transforms.RandomErasing(scale=(0.02, 0.4), value='random'))
    else:
        rotate = transforms.RandomApply([transforms.RandomRotation((-35, 35))], p=.2)
        crop = transforms.RandomApply([transforms.RandomResizedCrop(224, scale=(.6, 1), ratio=(1, 1))], p=.5)
        color_jitter = transforms.RandomApply([transforms.ColorJitter(brightness=.25, hue=.15, saturation=.05)], p=.4)
        train_transforms = [rotate, crop, color_jitter, transforms.Scale(224), transforms.ToTensor()]
        print('Loading Human Masks')
        for filename in os.listdir(args.hso_mask_dir):
            human_masks.append(cv2.resize(cv2.imread(os.path.join(args.hso_mask_dir, filename)), (224, 224)))
            means.append(np.mean(human_masks[-1] / 255))
        human_masks = np.stack(human_masks)
        human_masks = np.transpose(human_masks, (0, 3, 1, 2))
        human_masks[human_masks == 0] = 1
        human_masks[human_masks > 1] = 0

    # train_transforms.append(normalize)
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose(train_transforms)),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, human_masks=human_masks)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': 'resnet18',
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, args.occlusion_method)


def generate_cutout_mask(data, num_masks, mask_length):

    # code for cutout

    h, w = data.shape[2], data.shape[3]
    mask = np.ones(shape=(h, w))
    for _ in range(num_masks):
        x_coordinate = np.random.randint(0, w)
        y_coordinate = np.random.randint(0, h)
        mask[x_coordinate: x_coordinate + mask_length, y_coordinate: y_coordinate + mask_length] = 0
    return data * torch.tensor(mask, dtype=torch.float)


def generate_random_masks(data, percentage):

    # code for Irregular shaped masks

    h, w = data.shape[2], data.shape[3]
    mask = np.ones(shape=(h, w))
    local_mask_area = .05 * h * w
    sqr_root_local_mask_area = local_mask_area ** (1 / 2)
    while np.mean(mask) > 1 - percentage:
        local_mask_width = np.random.randint(low=int(sqr_root_local_mask_area / 2),
                                             high=int(sqr_root_local_mask_area * (3 / 2)))
        local_mask_height = int(local_mask_area / local_mask_width)
        top_left_pixel_y, top_left_pixel_x = np.random.randint(0, h - local_mask_height), np.random.randint(0, w - local_mask_width)
        mask[top_left_pixel_y: top_left_pixel_y + local_mask_height,
        top_left_pixel_x: top_left_pixel_x + local_mask_width] = 0
    return data * torch.tensor(mask, dtype=torch.float)


def human_masks_batch(data, human_masks):

    # Code for generating human-shaped occlusions

    n = data.shape[0]
    selected_masks = human_masks[np.random.choice(len(human_masks), n), :, :, :]
    binomial = np.reshape(np.random.binomial(1, .5, size=n), (n, 1, 1, 1))
    binomial = np.tile(binomial, (1, 3, data.shape[2], data.shape[3]))
    augmentation = selected_masks * binomial
    augmentation[augmentation == 0] = 2
    augmentation[augmentation == 1] = 0
    augmentation[augmentation == 2] = 1
    return data * torch.tensor(augmentation, dtype=torch.float)


def train(train_loader, model, criterion, optimizer, epoch, human_masks=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        if args.train_method == 'cutout' or args.train_method == 'irreg':
            if args.train_method == 'cutout':
                masked_input = generate_cutout_mask(input, args.num_masks, args.mask_length)
            else:
                masked_input = generate_random_masks(input, args.percent_masked)
            input = torch.cat([input, masked_input])
            target = target.repeat(2)

        elif args.train_method == 'hso':
            input = human_masks_batch(input, human_masks)

        data_time.update(time.time() - end)

        target = target.cuda()

        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, method):
    save_dir = 'places2/weights'
    torch.save(state, os.path.join(save_dir, 'resnet18_{}_latest.pth.tar'.format(method)))
    if is_best:
        shutil.copyfile(os.path.join(save_dir, 'resnet18_{}_latest.pth.tar'.format(method)),
        os.path.join(save_dir, 'resnet18_{}_best.pth.tar'.format(method)))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
