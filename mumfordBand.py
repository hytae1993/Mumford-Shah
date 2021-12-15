from __future__ import print_function
import os
import math
from numpy.core.numeric import False_

import torch
import torch.backends.cudnn as cudnn
import torchvision
from torchvision.utils import save_image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models

from model.ResNet import *
# from model.UNet import *
from model.autoEncoder import *  
from util.loss import Loss
from util.utils import *
from util.progress_bar import progress_bar
from util.scheduler_learning_rate import *
from util.laplace import * 

import numpy as np

from plot.plotMumfordBand import plot

class mumfordBand (object):
    def __init__(self, config, training_loader, val_loader):
        super(mumfordBand , self).__init__()
        self.CUDA = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.CUDA else 'cpu')
        self.seed = config.seed
        self.nEpochs = config.epoch
        self.lr = config.lr
        self.ms = config.ms
        self.mr = config.mr
        self.mt = config.mt
        self.loss = Loss()
        self.lapalce = Laplace(device=self.device).diffuse
        self.num_class = len(config.classes)
        self.log_interval = config.log
        self.config = config
        
        self.train_loader = training_loader
        self.val_loader = val_loader

        self.encoder = None
        self.maskDecoder = None
        self.classifier = None
        self.foreDecoder = None
        self.backDecoder = None

        self.plot = None

        self.optimizer = {}

        self.crossCriterion = None

        self.train_loss = []
        self.val_loss = []

    def build_model(self):
        self.encoder = Encoder(n_channels=3, n_classes=self.num_class)
        self.maskDecoder = DecoderSeg(n_channels=3, n_classes=3)

        self.encoder = self.encoder.to(self.device)
        self.maskDecoder = self.maskDecoder.to(self.device)

        self.plot = plot(self.train_loader, self.val_loader, self.encoder, self.maskDecoder,\
         self.device, self.config)

        self.crossCriterion = torch.nn.CrossEntropyLoss()

        if self.CUDA:
            cudnn.benchmark = True
            self.crossCriterion.cuda()

        self.optimizer['maskDecoder'] = torch.optim.SGD(self.maskDecoder.parameters(), lr=self.lr)
        self.optimizer['encoder'] = torch.optim.SGD(self.encoder.parameters(), lr=self.lr)

    def run(self, epoch, data_loader, work):
        if work == 'train':
            self.maskDecoder.train()
            self.encoder.train()
        elif work == 'val':
            self.maskDecoder.eval()
            self.encoder.eval()

        lossList = []
        maskRegionRegular = 0
        maskSmoothRegular = 0
        foregroundLoss = 0
        backgroundLoss = 0
        foreRegular = 0
        backRegular= 0 
        segloss = 0

        iter = 0
        num_data = 0

        for batch_num, (input, labelTarget) in enumerate(data_loader):
            iter += 1
            num_data += input.size(0)
            input = input.to(self.device)
            labelTarget = labelTarget.to(self.device).squeeze()
            
            if work == 'train':
                latent = self.encoder(input)
                mask = self.maskDecoder(latent)
                th_mask = get_threshold_mask(mask)
                dilate, erode = dilation(th_mask, kernel_size=3), erosion(th_mask, kernel_size=3)
                band = dilate + erode - 1

                foreground = dilate * input
                background = erode * input

                foregroundSmooth, _ = self.lapalce(foreground, dilate, num_iter=self.config.iter)
                backgroundSmooth,_ = self.lapalce(background, erode, num_iter=self.config.iter)

                # total variation for smooth, L1 loss for area of region
                maskSmooth = self.loss.tv(mask)
                # maskSmooth = self.loss.tv(th_mask)

                # calculate loss with non-thresholded mask
                mumfordLoss, foreSegLoss, backSegLoss = self.loss.segmentSmoothLossBand(input, mask, foregroundSmooth, backgroundSmooth, band)
                # calculate loss with thresholded mask
                # mumfordLoss, foreSegLoss, backSegLoss = self.loss.segmentSmoothLossBand(input, th_mask, foregroundSmooth, backgroundSmooth, band)
                regularization = self.ms * maskSmooth 

                segLoss = mumfordLoss
                totalLoss = segLoss + regularization
                

                self.optimizer['encoder'].zero_grad()
                self.optimizer['maskDecoder'].zero_grad()
                totalLoss.backward()
                self.optimizer['encoder'].step()
                self.optimizer['maskDecoder'].step()

            elif work == 'val':
                with torch.no_grad():
                    latent = self.encoder(input)
                    mask = self.maskDecoder(latent)
                    th_mask = get_threshold_mask(mask)
                    dilate, erode = dilation(th_mask, kernel_size=3), erosion(th_mask, kernel_size=3)
                    band = dilate + erode - 1

                    foreground = dilate * input
                    background = erode * input

                    foregroundSmooth, _ = self.lapalce(foreground, dilate, num_iter=self.config.iter)
                    backgroundSmooth, _ = self.lapalce(background, erode, num_iter=self.config.iter)

                    # total variation for smooth, L1 loss for area of region
                    maskSmooth = self.loss.tv(th_mask)

                    mumfordLoss, foreSegLoss, backSegLoss = self.loss.segmentSmoothLossBand(input, th_mask, foregroundSmooth, backgroundSmooth, band)

                    regularization = self.ms * maskSmooth 

                    segLoss = mumfordLoss
                    totalLoss = segLoss + regularization 

            maskSmoothRegular += (maskSmooth.item() * input.size(0))
            foregroundLoss += (foreSegLoss.item() * input.size(0))
            backgroundLoss += (backSegLoss.item() * input.size(0))
            segloss += (segLoss.item() * input.size(0))
            lossList.append(totalLoss.item())

            progress_bar(batch_num, len(data_loader))

        return np.mean(lossList), np.std(lossList), maskSmoothRegular/num_data, \
                    foregroundLoss/num_data, backgroundLoss/num_data, foreRegular/num_data, backRegular/num_data
                    
        
    def runner(self):

        for i in range(7):
            self.train_loss.append([])
            self.val_loss.append([])
       
        self.build_model()
        
        # visualize initialize data
        self.plot.plotResult(epoch=0, trainResult=None, valResult=None)
        
        # scheduler = scheduler_learning_rate_sigmoid_double(self.optimizer, self.nEpochs, [0.01, 0.1], [0.1, 0.00001], [10, 10], [0,0])

        for epoch in range(1, self.nEpochs + 1):
            print("\n===> Epoch {} starts:".format(epoch))
            # scheduler.step()

            trainResult = self.run(epoch, self.train_loader, 'train')
            valResult = self.run(epoch, self.val_loader, 'val')

            for i in range(7):
                self.train_loss[i].append(trainResult[i])
                self.val_loss[i].append(valResult[i])
            

            if epoch % self.log_interval == 0 or epoch == 1:
                self.plot.plotResult(epoch,self.train_loss, self.val_loss)

            if epoch == self.nEpochs:
                self.plot.plotResult(epoch, self.train_loss, self.val_loss)
                