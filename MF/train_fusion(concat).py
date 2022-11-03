
import os
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.nn import CosineEmbeddingLoss

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models.vgg import *
from conf import settings
from utils import get_network,  WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights
from dataset import CESM

def train(epoch):
    start = time.time()
    train_loss = 0.0 # cost function error
    correct = 0.0
    net.train()
    net3.train()
    p=0
    for i, x in enumerate(CESM_10_train_l):
        YY1 = x['LOW_ENERGY_CCL']
        YY2 = x['RECOMBINED_CCL']
        YY3 = x['LOW_ENERGY_CCR']
        YY4 = x['RECOMBINED_CCR']
        YY5 = x['LOW_ENERGY_MLOL']
        YY6 = x['RECOMBINED_MLOL']
        YY7 = x['LOW_ENERGY_MLOR']
        YY8 = x['RECOMBINED_MLOR']
        labels1=x['label1']
        labels2 = x['label2']
        labels3 = x['label3']
        labels4 = x['label4']

        YY_CCL = torch.cat((YY1, YY2), dim=1)
        YY_CCR = torch.cat((YY3, YY4), dim=1)
        YY_MLOL = torch.cat((YY5, YY6), dim=1)
        YY_MLOR = torch.cat((YY7, YY8), dim=1)
        # YY_CCL = YY1
        # YY_CCR = YY3
        # YY_MLOL =YY5
        # YY_MLOR =YY7



        labels1 = torch.IntTensor(labels1).to(torch.long)
        labels2 = torch.IntTensor(labels2).to(torch.long)

        if args.gpu:

            YY_CCL = YY_CCL.cuda()
            YY_CCR = YY_CCR.cuda()
            YY_MLOL = YY_MLOL.cuda()
            YY_MLOR = YY_MLOR.cuda()
            labels1=labels1.cuda()
            labels2 = labels2.cuda()

        optimizer.zero_grad()
        outputs_CCL,ou_CCL = net(YY_CCL)
        outputs_CCR,ou_CCR = net(YY_CCR)
        outputs_MLOL,ou_MLOL = net(YY_MLOL)
        outputs_MLOR,ou_MLOR = net(YY_MLOR)

        outputs1 = torch.cat((outputs_CCL, outputs_CCR), dim=1)
        outputs3 = torch.cat((outputs_MLOL, outputs_MLOR), dim=1)
        outputs2 = torch.cat((outputs_CCR,outputs_CCL), dim=1)
        outputs4 = torch.cat((outputs_MLOR, outputs_MLOL), dim=1)

        outputs11=torch.cat((outputs1, outputs3), dim=1)
        outputs22 = torch.cat((outputs2, outputs4), dim=1)
        ou11=net3(outputs11)
        ou22 = net3(outputs22)
        loss1 = loss_function(ou11, labels1)
        loss2 = loss_function(ou22, labels2)
        loss = loss1 + loss2
        print('loss1:{},loss2:{}'.format(
            loss1.item(),loss2.item()
        )

        )



        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, preds = ou11.max(1)
        _1, preds2 = ou22.max(1)
        correct += preds.eq(labels1).sum()
        correct += preds2.eq(labels2).sum()
        n_iter = (epoch - 1) * len(CESM_10_train_l) + i + 1



        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            # correct.float() / len(CESMdata),
            epoch=epoch,
            trained_samples=i * args.b + len(YY_CCL),
            total_samples=len(CESMdata)
        ))


        if epoch <= args.warm:
            warmup_scheduler.step()


    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))
    print('Average loss: {:.4f}\tAccuarcy: {:0.6f}'.format(
        train_loss / len(CESMdata),
        correct.float() / (2*len(CESMdata))
        ))


@torch.no_grad()
def eval_training(epoch=0, tb=True):

    start = time.time()
    net.eval()
    net3.eval()
    test_loss = 0.0  # cost function error
    correct = 0.0
    correct2=0.0

    for i, x in enumerate(CESM_10_test_l):

        YY1 = x['LOW_ENERGY_CCL']
        YY2 = x['RECOMBINED_CCL']
        YY3 = x['LOW_ENERGY_CCR']
        YY4 = x['RECOMBINED_CCR']
        YY5 = x['LOW_ENERGY_MLOL']
        YY6 = x['RECOMBINED_MLOL']
        YY7 = x['LOW_ENERGY_MLOR']
        YY8 = x['RECOMBINED_MLOR']
        labels1=x['label1']
        labels2 = x['label2']
        labels3 = x['label3']
        labels4 = x['label4']


        YY_CCL = torch.cat((YY1, YY2), dim=1)
        YY_CCR = torch.cat((YY3, YY4), dim=1)
        YY_MLOL = torch.cat((YY5, YY6), dim=1)
        YY_MLOR = torch.cat((YY7, YY8), dim=1)

        # YY_CCL = YY1
        # YY_CCR = YY3
        # YY_MLOL =YY5
        # YY_MLOR =YY7


        labels1 = torch.IntTensor(labels1).to(torch.long)
        labels2 = torch.IntTensor(labels2).to(torch.long)

        if args.gpu:

            YY_CCL = YY_CCL.cuda()
            YY_CCR = YY_CCR.cuda()
            YY_MLOL = YY_MLOL.cuda()
            YY_MLOR = YY_MLOR.cuda()
            labels1=labels1.cuda()
            labels2 = labels2.cuda()

        optimizer.zero_grad()
        outputs_CCL,ou_CCL = net(YY_CCL)
        outputs_MLOL, ou_MLOL = net(YY_MLOL)
        outputs_CCR,ou_CCR = net(YY_CCR)
        outputs_MLOR, ou_MLOR = net(YY_MLOR)


        outputs1 = torch.cat((outputs_CCL, outputs_CCR), dim=1)
        outputs3 = torch.cat((outputs_MLOL, outputs_MLOR), dim=1)
        outputs2 = torch.cat((outputs_CCR,outputs_CCL), dim=1)
        outputs4 = torch.cat((outputs_MLOR, outputs_MLOL), dim=1)
        outputs11=torch.cat((outputs1, outputs3), dim=1)
        outputs22 = torch.cat((outputs2, outputs4), dim=1)
        ou11=net3(outputs11)
        ou22 = net3(outputs22)
        loss1 = loss_function(ou11, labels1)
        loss2 = loss_function(ou22, labels2)
        loss = loss1 + loss2


        n_iter = (epoch - 1) * len(CESM_10_test_l) + i + 1
        test_loss += loss.item()
        _, preds = ou11.max(1)
        _1, preds2 = ou22.max(1)
        correct += preds.eq(labels1).sum()
        correct += preds2.eq(labels2).sum()

    finish = time.time()

    if args.gpu:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f} Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(CESMdata2),
        correct.float() / (2*len(CESMdata2)),
        finish - start
    ))
    print()


    return correct.float() / (2*len(CESMdata2))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
    parser.add_argument('-b', type=int, default=2, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=0, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    args = parser.parse_args()

    net = vgg16_bn()

    net = net.cuda()

    # path1 = 'pretrained models'
    # net.load_state_dict(torch.load(path1),strict=False)
    net3=nn.Linear(2048,2)
    net3.cuda()





    CESMdata = CESM(base_dir='./h5py/train',transform=transforms.Compose([
                       # Random表示有可能做，所以也可能不做
                       transforms.RandomHorizontalFlip(p=0.5),# 水平翻转
                       transforms.RandomVerticalFlip(p=0.5), # 上下翻转
                       transforms.RandomRotation(10), # 随机旋转-10°~10°
                       # transforms.RandomRotation([90, 180]), # 随机在90°、180°中选一个度数来旋转，如果想有一定概率不旋转，可以加一个0进去
                       # transforms.ColorJitter(brightness=0.3, contrast=0.3, hue=0.3)
                       # transforms.Normalize(0, 1)
                       transforms.ToTensor(),

                   ]))
    CESM_10_train_l = DataLoader(CESMdata, batch_size=args.b, shuffle=True, drop_last=False,
                                 pin_memory=torch.cuda.is_available())

    CESMdata2 = CESM(base_dir='./h5py/valid',transform=transforms.Compose([
                       transforms.ToTensor(),
                   ]))
    CESM_10_test_l = DataLoader(CESMdata2, batch_size=8, shuffle=False, drop_last=False,
                                 pin_memory=torch.cuda.is_available())






    loss_function = nn.CrossEntropyLoss()
    loss_function.cuda()

    optimizer = optim.SGD([{"params":net.parameters()},{"params":net3.parameters()}], lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.5) #learning rate decay
    iter_per_epoch = len(CESM_10_train_l)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    if args.resume:
        recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, args.net), fmt=settings.DATE_FORMAT)
        if not recent_folder:
            raise Exception('no recent folder were found')

        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, 'vgg16', recent_folder)

    else:
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, 'vgg16', settings.TIME_NOW)



    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    best_acc2=0.0
    if args.resume:
        best_weights = best_acc_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if best_weights:
            weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, best_weights)
            print('found best acc weights file:{}'.format(weights_path))
            print('load best training file to test acc...')
            net.load_state_dict(torch.load(weights_path))
            best_acc = eval_training(tb=False)
            print('best acc is {:0.2f}'.format(best_acc))

        recent_weights_file = most_recent_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if not recent_weights_file:
            raise Exception('no recent weights file were found')
        weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, recent_weights_file)
        print('loading weights file {} to resume training.....'.format(weights_path))
        net.load_state_dict(torch.load(weights_path))

        resume_epoch = last_epoch(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))


    for epoch in range(1, settings.EPOCH + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        if args.resume:
            if epoch <= resume_epoch:
                continue


        train(epoch)
        acc= eval_training(epoch)

        #start to save best performance model after learning rate decay to 0.01
        if epoch <= settings.MILESTONES[3] and best_acc < acc:
            weights_path = checkpoint_path.format(net='vgg16', epoch=epoch, type='best')
            weights_path3 = checkpoint_path.format(net='net3', epoch=epoch, type='best')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
            torch.save(net3.state_dict(), weights_path3)
            best_acc = acc

            continue






