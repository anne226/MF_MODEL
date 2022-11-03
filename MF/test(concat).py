#test.py
#!/usr/bin/env python3

""" test neuron network performace
print top1 and top5 err on test dataset
of a model

author baiyu
"""

import argparse

from matplotlib import pyplot as plt

import torch
import torchvision.transforms as transforms
from sklearn import metrics
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from torch.utils.data import DataLoader
from xlwt import *
from conf import settings
from utils import get_network
from dataset import CESM
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

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models.vgg import *
from conf import settings
from utils import get_network,  WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights
from dataset import CESM



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('-net', type=str, required=True, help='net type')
    # parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    # parser.add_argument('-weights2', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
    parser.add_argument('-b', type=int, default=1, help='batch size for dataloader')
    args = parser.parse_args()

    net = vgg16_bn()

    net.cuda()

    net3=nn.Linear(2048,2)
    net3.cuda()

    CESMdata2 = CESM(base_dir=r'./h5py/test',transform=transforms.Compose([
                       transforms.ToTensor(),
                   ]))

    CESM_10_test_l = DataLoader(CESMdata2, batch_size=1, shuffle=False, drop_last=True,
                                 pin_memory=torch.cuda.is_available())

    path1 = '/path to weights'
    path2 = '/path to weights_net3'



    net.load_state_dict(torch.load(path1))

    net3.load_state_dict(torch.load(path2))
    # net3.eval()

    # print(net)
    net.eval()
    net3.eval()
    test_loss = 0.0 # cost function error
    correct = 0.0
    correct2=0.0
    class_correct =list(0.for i in range(2))
    class_total = list(0.for i in range(2))
    correct_1 = 0.0
    correct_5 = 0.0
    total = 0
    tp_l = 0.0
    tn_l = 0.0
    fp_l = 0.0
    fn_l = 0.0
    tp_r = 0.0
    tn_r = 0.0
    fp_r = 0.0
    fn_r = 0.0
    pro=[]
    label=[]
    with torch.no_grad():
        for i, x in enumerate(CESM_10_test_l):

            YY1 = x['LOW_ENERGY_CCL']
            YY2 = x['RECOMBINED_CCL']
            YY3 = x['LOW_ENERGY_CCR']
            YY4 = x['RECOMBINED_CCR']
            YY5 = x['LOW_ENERGY_MLOL']
            YY6 = x['RECOMBINED_MLOL']
            YY7 = x['LOW_ENERGY_MLOR']
            YY8 = x['RECOMBINED_MLOR']
            labels1 = x['label1']
            labels2 = x['label2']
            labels3 = x['label3']
            labels4 = x['label4']
            YY_CCL = torch.cat((YY1, YY2), dim=1)
            YY_CCR = torch.cat((YY3, YY4), dim=1)
            YY_MLOL = torch.cat((YY5, YY6), dim=1)
            YY_MLOR = torch.cat((YY7, YY8), dim=1)

            # YY_CCL = YY1
            # YY_CCR = YY3
            # YY_MLOL = YY5
            # YY_MLOR = YY7


            label.append(labels1.cpu().item())
            label.append(labels2.cpu().item())


            if args.gpu:

                YY_CCL = YY_CCL.cuda()
                YY_CCR = YY_CCR.cuda()
                YY_MLOL = YY_MLOL.cuda()
                YY_MLOR = YY_MLOR.cuda()
                labels1 = labels1.cuda()
                labels2 = labels2.cuda()


            outputs_CCL, ou_CCL = net(YY_CCL)
            outputs_MLOL, ou_MLOL = net(YY_MLOL)
            outputs_CCR, ou_CCR = net(YY_CCR)
            outputs_MLOR, ou_MLOR = net(YY_MLOR)
            outputs1 = torch.cat((outputs_CCL, outputs_CCR), dim=1)
            outputs3 = torch.cat((outputs_MLOL, outputs_MLOR), dim=1)
            outputs2 = torch.cat((outputs_CCR, outputs_CCL), dim=1)
            outputs4 = torch.cat((outputs_MLOR, outputs_MLOL), dim=1)
            outputs11 = torch.cat((outputs1, outputs3), dim=1)
            outputs22 = torch.cat((outputs2, outputs4), dim=1)
            ou11 = net3(outputs11)
            ou22 = net3(outputs22)


            sot1=torch.softmax(ou11,dim=1)
            sot2 = torch.softmax(ou22, dim=1)


            pro1=torch.index_select(sot1.cpu(),dim=1,index=torch.tensor(1))
            pro2 = torch.index_select(sot2.cpu(),dim=1,index=torch.tensor(1))
            pro.append(pro1.cpu().item())
            pro.append(pro2.cpu().item())

            _, preds = ou11.max(1)
            _1, preds2 = ou22.max(1)
            correct += preds.eq(labels1).sum()
            correct += preds2.eq(labels2).sum()










            tp_l += torch.sum(np.logical_and(preds.cpu() == 1, labels1.cpu() == 1))
            tn_l += torch.sum(np.logical_and(preds.cpu() == 0, labels1.cpu() == 0))
            fp_l += torch.sum(np.logical_and(preds.cpu() == 1, labels1.cpu() == 0))
            fn_l += torch.sum(np.logical_and(preds.cpu() == 0, labels1.cpu() == 1))


            tp_r += torch.sum(np.logical_and(preds2.cpu() == 1, labels2.cpu() == 1))
            tn_r += torch.sum(np.logical_and(preds2.cpu() == 0, labels2.cpu() == 0))
            fp_r += torch.sum(np.logical_and(preds2.cpu() == 1, labels2.cpu() == 0))
            fn_r += torch.sum(np.logical_and(preds2.cpu() == 0, labels2.cpu()== 1))


        recall_l = tp_l / (tp_l + fn_l)
        recall_r = tp_r / (tp_r + fn_r)

        # 计算precision
        precision_l = tp_l / (tp_l + fp_l)
        precision_r = tp_r / (tp_r + fp_r)
        npv_l = tn_l / (tn_l + fn_l)
        npv_r=tn_r / (tn_r + fn_r)
        # 计算f1-score
        f1_score_l = 2 * recall_l * precision_l / (recall_l + precision_l)
        f1_score_r = 2 * recall_r * precision_r / (recall_r + precision_r)
        # 计算falseAlarmRate
        spe_l = tn_l / (tn_l+ fp_l)
        spe_r = tn_r / (tn_r + fp_r)
        false_alarm_rate_l = fp_l / (tn_l + fp_l)
        false_alarm_rate_r = fp_r / (tn_r + fp_r)
        auc_score = roc_auc_score(label, pro)

        print(auc_score)

        print(
            ' Recall_L: {:.4f}\tPrecision_L: {:.4f}\tNpv_L: {:.4f}\tSpecificity_L:{:.4f}\tF1_score_L: {:.4f}\tFalse_alarm_rate_L: {:.4f}'.format(
                (recall_l+recall_r)/2, (precision_l+precision_r)/2,(npv_l+npv_r)/2,(spe_l+spe_r)/2,(f1_score_l+f1_score_r)/2,
                (false_alarm_rate_l+false_alarm_rate_r)/2
            ))

        # 计算accuracy
        print('Test set: Accuracy: {:.4f}'.format(
            correct.float() / (2*len(CESMdata2))))

        precision1, recall1, thresholds1 = precision_recall_curve(label, pro)
        auc_precision_recall = auc(recall1, precision1)
        print(auc_precision_recall)
        fpr, tpr, threshold = metrics.roc_curve(label, pro)
        roc_auc = metrics.auc(fpr, tpr)
        plt.figure(figsize=(6, 6))
        plt.title('Validation ROC')
        plt.plot(fpr, tpr, 'b', label='Val AUC = %0.3f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()


        plt.figure(figsize=(6, 6))
        plt.title('Validation ROC')
        plt.plot(precision1, recall1, 'b', label='Val PRAUC = %0.3f' % auc_precision_recall)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('Precision')
        plt.xlabel('Recall')
        plt.show()



