import torch
import numpy as np
import torch.nn as nn
from util.laplace import *
from util.laplaceSum import *
import kornia.morphology as mp
import matplotlib.pyplot as plt
import os

def find_jaccard_overlap(set_1, set_2, eps=1e-5):
    """
    Find the Jaccard Overlap (IoU) of every mask between two sets of segmentation masks.
    :param set_1: set 1, a tensor of dimensions (1, 224, 224)
    :param set_2: set 2, a tensor of dimensions (1, 224, 224)
    :return: Jaccard Overlap in set 1 with respect to each of the boxes in set 2
    """
    # Find intersections
    intersection = np.logical_and(set_1, set_2)
    # Find the union
    union = np.logical_or(set_1, set_2)
    return intersection / union

def get_image(input, encoder, maskDecoder):
    """
    Get the mask
    :param input: input image
    :param encoder: encoder
    :param maskDecoder: decoder
    :return mask: the mask which bi-partition the input image to the foregroun and the background
    """
    with torch.no_grad():
        layer = encoder(input)
        mask = maskDecoder(layer)
        # th_mask = get_threshold_mask(mask)
        # f, b = Laplace().diffuse(input, th_mask, num_iter=1)
        # f, b = f.cpu(), b.cpu()
        mask = mask.cpu()
        
        return mask

def get_mum_image(input, encoder, maskDecoder, iter):
    with torch.no_grad():
        layer = encoder(input)
        mask = maskDecoder(layer)
        th_mask = get_threshold_mask(mask)
        dilate, erode = dilation(th_mask, kernel_size=3), erosion(th_mask, kernel_size=3)
        band = dilate + erode - 1
        foreground = dilate * input
        background = erode * input
        foregroundSmooth, _ = Laplace().diffuse(foreground, dilate, num_iter=iter)
        _, backgroundSmooth = Laplace().diffuse(background, erode, num_iter=iter)
        
        return mask.cpu(), dilate.cpu(), erode.cpu(), band.cpu(), foreground.cpu(), background.cpu(), foregroundSmooth.cpu(), backgroundSmooth.cpu()

def get_mum_image2(input, encoder, maskDecoder, decoder):
    with torch.no_grad():
        layer = encoder(input)
        mask = maskDecoder(layer)
        fore, back = decoder(layer)
        mask = mask.cpu()
        fore, back = fore.cpu(), back.cpu()

        return mask, fore, back

def get_mum_image_band(input, encoder, maskDecoder, iter):
    with torch.no_grad():
        layer = encoder(input)
        mask = maskDecoder(layer)
        th_mask = get_threshold_mask(mask)
        dilate, erode = dilation(th_mask, kernel_size=3), erosion(th_mask, kernel_size=3)
        band = dilate + erode - 1

        foreground = dilate * input
        background = erode * input

        foregroundSmooth, _ = Laplace().diffuse(foreground, dilate, num_iter=iter)
        backgroundSmooth,_ = Laplace().diffuse(background, erode, num_iter=iter)

        return mask.cpu(), dilate.cpu(), erode.cpu(), band.cpu(), foreground.cpu(), background.cpu(), foregroundSmooth.cpu(), backgroundSmooth.cpu()

def get_mum_image_band_multi(input, encoder, maskDecoder, iter, last):
    with torch.no_grad():
        ma, th, er, dil, fore, ba = [], [], [], [], [], []
        layer = encoder(input)
        mask = maskDecoder(layer)
        for i in range(last):
            mask_ = mask[:,i,:,:].unsqueeze(dim=1)
            th_mask = get_threshold_mask(mask_)
            dilate, erode = dilation(th_mask, kernel_size=3), erosion(th_mask, kernel_size=3)
            band = dilate + erode - 1

            foreground = dilate * input
            foregroundSmooth, _ = Laplace().diffuse(foreground, dilate, num_iter=iter)
            th.append(th_mask.cpu())
            er.append(erode.cpu())
            dil.append(dilate.cpu())
            ba.append(band.cpu())
            fore.append(foregroundSmooth.cpu())
            ma.append(mask_.cpu())

        return ma, th, er, dil, ba, fore

def get_mum_image_band_sum(input, encoder, maskDecoder):
    with torch.no_grad():
        layer = encoder(input)
        mask = maskDecoder(layer)
        th_mask = get_threshold_mask(mask)
        dilate, erode = dilation(th_mask, kernel_size=3), erosion(th_mask, kernel_size=3)
        band = dilate + erode - 1

        foreground = dilate * input
        background = erode * input

        foregroundSmoothList = LaplaceSum().diffuse(foreground, dilate)
        backgroundSmoothList = LaplaceSum().diffuse(background, erode)

        return mask.cpu(), dilate.cpu(), erode.cpu(), band.cpu(), foreground.cpu(), background.cpu(), foregroundSmoothList, backgroundSmoothList


def get_threshold_mask(mask):
    """
    Threshold the mask based on 0.5.
    :param mask: input mask
    :return: thresholded mask
    """
    # with torch.no_grad():
    out = (mask>0.5).float()

    out = nn.Sigmoid()(mask * 50)

    m = torch.nn.Threshold(0.5, 0)
    # b = m(mask)
    # d = -1 * b
    # n = torch.nn.Threshold(-0.000001, 1)
    # out = n(d)

    return out

def get_mean(mask, fore, back):
    """
    get mean value of smoothed region
    mean_f = sum(fore * mask) / sum(mask)
    mean_b = sum(back * (1-mask)) / sum(1-mask)
    :param mask: binary mask
    :param fore: foreground smoothed matrix by laplace
    :param back: background smoothed matrix by laplace
    """
    # pixel-wise constant segmentation loss
    foreground = mask * fore
    background = (1-mask) * back
    foregroundMean = foreground.sum(dim=2,keepdim=True).sum(dim=3,keepdim=True) / mask.sum(dim=2,keepdim=True).sum(dim=3,keepdim=True)
    backgroundMean = background.sum(dim=2,keepdim=True).sum(dim=3,keepdim=True) / (1-mask).sum(dim=2,keepdim=True).sum(dim=3,keepdim=True)
    return foregroundMean, backgroundMean

def erosion(img, kernel_size):
    kernel = torch.ones((kernel_size, kernel_size)).cuda()
    erode = mp.erosion(img, kernel)
    erode = 1 - erode
    return erode

def dilation(img, kernel_size):
    kernel = torch.ones((kernel_size, kernel_size)).cuda()
    dilate = mp.dilation(img, kernel)
    return dilate

def saveModel(encoder, maskDecoder, args, work):
    try:
        torch.save(maskDecoder.state_dict(), './savedModel/{}/dogCat_maskDecoder_ms_{}'.format(work, args.ms))
        # torch.save(foreDecoder.state_dict(), './savedModel/Chan_vese/dogCat_foreDecoder_mr_{}_ms_{}_ir_{}'.format(args.mr, args.ms, args.ir))
        # torch.save(backDecoder.state_dict(), './savedModel/Chan_vese/dogCat_backDecoder_mr_{}_ms_{}_ir_{}'.format(args.mr, args.ms, args.ir))
        torch.save(encoder.state_dict(), './savedModel/{}/dogCat_encoder__ms_{}'.format(work, args.ms))
    except FileNotFoundError:
        os.makedirs('./savedModel/{}'.format(work))
        torch.save(maskDecoder.state_dict(), './savedModel/{}/dogCat_maskDecoder_ms_{}'.format(work, args.ms))
        # torch.save(foreDecoder.state_dict(), './savedModel/Chan_vese/dogCat_foreDecoder_mr_{}_ms_{}_ir_{}'.format(args.mr, args.ms, args.ir))
        # torch.save(backDecoder.state_dict(), './savedModel/Chan_vese/dogCat_backDecoder_mr_{}_ms_{}_ir_{}'.format(args.mr, args.ms, args.ir))
        torch.save(encoder.state_dict(), './savedModel/{}/dogCat_encoder_ms_{}'.format(work, args.mr, args.ms))