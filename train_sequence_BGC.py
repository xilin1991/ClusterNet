from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
# import socket
import time
import argparse
import random
from datetime import datetime

import einops
import cv2
# import numpy as np
# from PIL import Image
import torch
# import torch.nn as nn
# import torch.nn.functional as F
from torch.utils import data
import torch.backends.cudnn as cudnn
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

from models.ClusterNet import ClusterNet_BGC
from datasets.data_loader import SequenceLoader
from loss.loss_helpers import ClusterLoss_BGC
from utils.average_meter import AverageMeter
from utils.helpers import cluster_iou, mask_assignment, mask_save, str2bool


DATE_FORMAT = '%b%d_%H-%M-%S'
TIME_NOW = datetime.now().strftime(DATE_FORMAT)
# AMP setting
USE_AMP = str2bool(str(os.environ['USE_AMP']))


def get_arguments():
    description = 'sequence training'
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--batch_size', default=8, type=int, help='batch size for training')
    parser.add_argument('--max_epoch', default=100, type=int, help='epoch of training')
    parser.add_argument('--workers', default=4, type=int, help='the number of workers')
    parser.add_argument('--lr', type=float, default=0.001)

    parser.add_argument('--n_input', type=int, default=3)
    parser.add_argument('--n_clusters', type=int, default=10)
    parser.add_argument('--iters', type=int, default=50)
    parser.add_argument('--n_z', type=int, default=10)
    parser.add_argument('--pretrain_path', type=str, default='./checkpoints')
    parser.add_argument('--results_path', type=str, default='./results_sequence')

    parser.add_argument('--data_dir', default=None, type=str, help='dataset root dir')
    parser.add_argument('--seq_name', default=None, type=str)
    parser.add_argument('--resolution', default=[480, 856], type=list, help='the resolution of image for training or inference')
    parser.add_argument('--to_rgb', action='store_true', help='whether flow to RGB image')
    parser.add_argument('--with_gt', action='store_true', help='whether load annotations')

    parser.add_argument('--display_session', default=20, type=int, help='display iterations for training')
    parser.add_argument('--threshold', default=0.5, type=float, help='threshold for background')

    parser.add_argument('--seed', type=int, default=304)

    return parser.parse_args()


def train(model, sample, optimizer, criterion, scaler, args_parser, device):
    iou_max = 0
    iou_total = AverageMeter()
    x = sample['flow'].to(device)
    y = sample['seg'][0].numpy()
    seq_name = sample['meta']['seq_name']
    img_name = sample['meta']['img_name']
    save_size = sample['meta']['ori_size']
    model.train()
    for it in range(args_parser.iters):
        optimizer.zero_grad()
        with autocast(enabled=USE_AMP):
            # start_time = time.time()
            x_bar, z, mask, preds = model(x)
            # end_time = time.time()
            # print('Running Time: {:.10f}'.format(end_time - start_time))
            loss = criterion(x_bar, x, z, mask)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        y_pred = preds['indexes'].data
        y_pred = y_pred.cpu().detach().numpy()
        assert y_pred.shape[0] == 1
        y_pred = y_pred[0]
        y_pred = cv2.resize(y_pred, (856, 480), interpolation=cv2.INTER_NEAREST)
        y_pred = einops.rearrange(y_pred, 'h w -> (h w)')

        y_pred = mask_assignment(y, y_pred, args_parser.n_clusters)

        iou = cluster_iou(y, y_pred)
        iou_total.update(iou)
        if iou >= iou_max:
            iou_max = iou
            mask = einops.rearrange(y_pred, '(h w) -> h w', h=480, w=856)
            mask_save(mask, save_dir=os.path.join(args_parser.results_path, *seq_name, '{}.png'.format(*img_name)), save_size=save_size)
        print('Iter {:4d}'.format(it), '| Current Acc {:.4f}'.format(iou), ' | Max Acc {:.4f}'.format(iou_max),
              ' | Loss {:.4f}'.format(loss.item()))

    return iou_max


def main():
    cudnn.enabled = True
    cudnn.benchmark = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: {}".format(device))

    args_parser = get_arguments()
    if args_parser.seed is not None:
        random.seed(args_parser.seed)
        torch.manual_seed(args_parser.seed)
    args_parser.cuda = torch.cuda.is_available()
    print(args_parser)

    if not os.path.exists(os.path.join(args_parser.pretrain_path)):
        os.makedirs(os.path.join(args_parser.pretrain_path))

    if not os.path.exists(os.path.join(args_parser.results_path)):
        os.makedirs(os.path.join(args_parser.results_path))

    # log_dir = os.path.join('./runs')
    # if not os.path.exists(log_dir):
    #     os.makedirs(log_dir)
    # log_dir = os.path.join(log_dir, '{0}_{1}'.format(TIME_NOW, socket.gethostname()))
    # writer = SummaryWriter(log_dir=log_dir)

    model = ClusterNet_BGC(arch=[64, 'MP', 128, 'MP', 256], n_input=args_parser.n_input,
                           n_z=args_parser.n_z, n_clusters=args_parser.n_clusters).to(device)

    dataset = SequenceLoader(root_dir=args_parser.data_dir,
                             seq_name=args_parser.seq_name,
                             resolution=args_parser.resolution,
                             to_rgb=args_parser.to_rgb,
                             with_gt=args_parser.with_gt)

    db_loader = data.DataLoader(dataset,
                                batch_size=args_parser.batch_size,
                                shuffle=True,
                                num_workers=args_parser.workers,
                                pin_memory=True,
                                drop_last=False)

    optimizer = optim.Adam(model.parameters(), lr=args_parser.lr)
    # import numpy as np
    # total_parameters = 0
    # model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    # total_parameters = sum([np.prod(p.size()) for p in model_parameters])
    # total_parameters = (1.0 * total_parameters / (1000 * 1000))
    # print('Total network parameters: ' + str(total_parameters) + ' million')
    scaler = GradScaler(enabled=USE_AMP)
    criterion = ClusterLoss_BGC(threshold=args_parser.threshold)

    # Pretraining
    pretrain_epochs = 100
    # start_time = time.time()
    if not os.path.exists(os.path.join(args_parser.pretrain_path, 'pretrain_{}.pth'.format(args_parser.seq_name))):
        model.train()
        for epoch in range(pretrain_epochs):
            total_loss = 0
            for it, sample in enumerate(db_loader):
                optimizer.zero_grad()
                x = sample['flow'].to(device)
                with autocast(enabled=USE_AMP):
                    x_bar, z, mask = model(x, pretrain_process=True)
                    loss = criterion(x_bar, x, z, mask)
                total_loss += loss.item()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            print("Epoch {:4d} loss={:.4f}".format(epoch, total_loss / (it + 1)))
            if epoch == pretrain_epochs - 1:
                # end_time = time.time()
                # print('Running Time: {:.1f}'.format(end_time - start_time))
                torch.save(model.state_dict(), os.path.join(args_parser.pretrain_path,
                                                            'pretrain_{}.pth'.format(args_parser.seq_name)))
        print("Model saved to {}.".format(os.path.join(args_parser.pretrain_path,
                                                       'pretrain_{}.pth'.format(args_parser.seq_name))))

    dataset = SequenceLoader(root_dir=args_parser.data_dir,
                             seq_name=args_parser.seq_name,
                             resolution=args_parser.resolution,
                             to_rgb=args_parser.to_rgb,
                             with_gt=args_parser.with_gt)

    db_loader = data.DataLoader(dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=args_parser.workers,
                                pin_memory=True,
                                drop_last=False)
    # seq_len = len(dataset)
    iou_avg = AverageMeter()
    results_dir = os.path.join(args_parser.results_path, args_parser.seq_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    for it, sample in enumerate(db_loader):
        seq_name = sample['meta']['seq_name']
        start_time = time.time()
        model.load_state_dict(torch.load(os.path.join(args_parser.pretrain_path, 'pretrain_{}.pth'
                                                      .format(args_parser.seq_name)), map_location=lambda storage, loc: storage))
        print('Load pre-trained model from {}'
              .format(os.path.join(args_parser.pretrain_path, 'pretrain_{}.pth'.format(args_parser.seq_name))))
        iou_max = train(model=model, sample=sample, optimizer=optimizer,
                        criterion=criterion, scaler=scaler, args_parser=args_parser, device=device)
        iou_avg.update(iou_max)
        end_time = time.time()
        print('Running Time: {:.1f}'.format(end_time - start_time))
    print('==> Sequence: {}'.format(*seq_name), ' | Avg ACC: {:.4f}'.format(iou_avg.avg))


if __name__ == '__main__':
    main()
