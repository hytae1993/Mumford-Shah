import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np



class Loss: 
    
    def __init__(self):
        self.area_loss_coef = 8
        self.smoothness_loss_coef = 0.5
        self.preserver_loss_coef = 0.3
        self.area_loss_power = 0.3
    
    def tv(self, image):
        x_loss = torch.mean((torch.abs(image[:,:,1:,:] - image[:,:,:-1,:])))
        y_loss = torch.mean((torch.abs(image[:,:,:,1:] - image[:,:,:,:-1])))

        return (x_loss + y_loss)

    def regionLoss(self, image):
        mask_mean = F.avg_pool2d(image, image.size(2), stride=1).squeeze().mean()

        return mask_mean

    def segmentConstantLoss(self, image, mask):
        # pixel-wise constant segmentation loss
        foreground = mask * image
        background = (1-mask) * image

        foregroundCenter = foreground.sum(dim=2,keepdim=True).sum(dim=3,keepdim=True) / mask.sum(dim=2,keepdim=True).sum(dim=3,keepdim=True)
        backgroundCenter = background.sum(dim=2,keepdim=True).sum(dim=3,keepdim=True) / (1-mask).sum(dim=2,keepdim=True).sum(dim=3,keepdim=True)

        foregroundLoss = (mask * ((image - foregroundCenter) ** 2)).mean()
        backgroundLoss = ((1-mask) * (image - backgroundCenter) ** 2).mean()

        # foregroundLoss = (((image - foregroundCenter)**2) * mask).sum() / image.numel()
        # backgroundLoss = (((image - backgroundCenter)**2) * (1 - mask)).sum() / image.numel()

        return foregroundLoss, backgroundLoss
  
    def segmentSmoothLoss(self, image, mask, foreCenter, backCenter):
        # pixel-wise smooth segmentation loss

        foregroundLoss = ((mask) * ((image - foreCenter)**2)).mean()
        backgroundLoss = ((1-mask) * ((image - backCenter)**2)).mean()

        return foregroundLoss, backgroundLoss

    def segmentSmoothLossBand(self, image, mask, foreSmooth, backSmooth, band):
        # pixel-wise smooth segmentation loss

        foregroundLoss = ((mask) * ((image - foreSmooth)**2))
        backgroundLoss = ((1-mask) * ((image - backSmooth)**2))

        segLoss = (foregroundLoss + backgroundLoss) * band
        segLoss = segLoss.sum() / band.sum()

        # return segLoss / mask.shape[0], (foregroundLoss*band).sum() / (band.sum() * mask.shape[0]),\
        #      (backgroundLoss*band).sum() / (band.sum() * mask.shape[0])

        return segLoss, (foregroundLoss*band).sum() / (band.sum()),\
             (backgroundLoss*band).sum() / (band.sum())

    def segmentSmoothLossBandMulti(self, image, mask, foreSmooth, band):
        # pixel-wise smooth segmentation loss

        foregroundLoss = ((mask) * ((image - foreSmooth)**2))

        segLoss = (foregroundLoss) * band
        segLoss = segLoss.sum() / band.sum()

        # return segLoss / mask.shape[0], (foregroundLoss*band).sum() / (band.sum() * mask.shape[0]),\
        #      (backgroundLoss*band).sum() / (band.sum() * mask.shape[0])

        return segLoss

    def segmentSmoothLossBandSum(self, image, mask, foreSmoothList, backSmoothList, band):
        # pixel-wise smooth segmentation loss

        foregroundLoss = torch.zeros_like(image).cuda()
        backgroundLoss = torch.zeros_like(image).cuda()
        for foreSmooth in foreSmoothList:
            foregroundLoss += foregroundLoss + ((mask) * ((image - foreSmooth)**2))
        for backSmooth in backSmoothList:
            backgroundLoss += ((1-mask) * ((image - backSmooth)**2))

        segLoss = (foregroundLoss + backgroundLoss) * band
        segLoss = segLoss.sum() / band.sum()

        # return segLoss / mask.shape[0], (foregroundLoss*band).sum() / (band.sum() * mask.shape[0]),\
        #      (backgroundLoss*band).sum() / (band.sum() * mask.shape[0])

        return segLoss, (foregroundLoss*band).sum() / (band.sum()),\
             (backgroundLoss*band).sum() / (band.sum())

    def segmentConstantAdversarialLoss(self, image, mask):
        # pixel-wise constant segmentation loss
        foreground = mask * image
        background = (1-mask) * image

        foregroundCenter = foreground.sum(dim=2,keepdim=True).sum(dim=3,keepdim=True) / mask.sum(dim=2,keepdim=True).sum(dim=3,keepdim=True)
        backgroundCenter = background.sum(dim=2,keepdim=True).sum(dim=3,keepdim=True) / (1-mask).sum(dim=2,keepdim=True).sum(dim=3,keepdim=True)

        foregroundLoss = (mask * ((image - foregroundCenter) ** 2)).mean()
        backgroundLoss = ((1-mask) * (image - backgroundCenter) ** 2).mean()

        # foregroundLoss = (((image - foregroundCenter)**2) * mask).sum() / image.numel()
        # backgroundLoss = (((image - backgroundCenter)**2) * (1 - mask)).sum() / image.numel()

        return foregroundLoss, backgroundLoss, foregroundCenter, backgroundCenter

