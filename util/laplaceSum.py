# This function solves a Laplance equation
# "Byung-Woo Hong in Chung-Ang University AI department"
#
# objective function:
#   | I - f |^2 * X + | I - b |^2 * (1 - X) + alpha * | \nabla f |^2 + alpha * | \nabla b |^2
#
# Euler-Lagrange equation:
#   - alpha \Delta u = I \cdot X - u \cdot X
#
# f     : foreground
# b     : background
# I     : input data
# X  : characteristic function (binary)
# alpha : weight for the regularization (L_2^2)
import numpy as np 
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


class LaplaceSum:

    def __init__(self, dt=0.25, alpha=1, device='cuda'):

        self.dt     = dt
        self.alpha  = alpha
        self.device = device


    def diffuse(self, I, X, num_iter=1):

        number_image_I  = I.shape[0]
        channel_image_I = I.shape[1]
        height_image_I  = I.shape[2]
        width_image_I   = I.shape[3]
        
        number_image_X  = X.shape[0]
        channel_image_X = X.shape[1]
        height_image_X  = X.shape[2]
        width_image_X   = X.shape[3]

        assert I.ndim == 4,         'dimension of input image I should be 4'
        assert X.ndim == 4,         'dimension of input mask X should be 4'
        assert I.ndim == X.ndim,    'dimension of input image I should match dimension of input mask X'

        p2d = (1, 1, 1, 1)

        _I_expand           = F.pad(I, p2d, 'replicate')
        _X_f_expand         = F.pad(X, p2d, 'replicate') 
        _X_b_expand         = F.pad(1-X, p2d, 'replicate')

        _X_f_expand_up      = torch.roll(_X_f_expand, -1, 2)
        _X_f_expand_down    = torch.roll(_X_f_expand, +1, 2)
        _X_f_expand_left    = torch.roll(_X_f_expand, -1, 3)
        _X_f_expand_right   = torch.roll(_X_f_expand, +1, 3)

        _X_b_expand_up      = torch.roll(_X_b_expand, -1, 2)
        _X_b_expand_down    = torch.roll(_X_b_expand, +1, 2)
        _X_b_expand_left    = torch.roll(_X_b_expand, -1, 3)
        _X_b_expand_right   = torch.roll(_X_b_expand, +1, 3)

        _X_f_border_up      = torch.mul(_X_f_expand, 1-_X_f_expand_up)
        _X_f_border_down    = torch.mul(_X_f_expand, 1-_X_f_expand_down)
        _X_f_border_left    = torch.mul(_X_f_expand, 1-_X_f_expand_left)
        _X_f_border_right   = torch.mul(_X_f_expand, 1-_X_f_expand_right)

        _X_b_border_up      = torch.mul(_X_b_expand, 1-_X_b_expand_up)
        _X_b_border_down    = torch.mul(_X_b_expand, 1-_X_b_expand_down)
        _X_b_border_left    = torch.mul(_X_b_expand, 1-_X_b_expand_left)
        _X_b_border_right   = torch.mul(_X_b_expand, 1-_X_b_expand_right)

        _f_expand           = torch.mul(_I_expand, _X_f_expand)
        _b_expand           = torch.mul(_I_expand, _X_b_expand)
        _f_update           = torch.zeros_like(_f_expand).to(self.device)
        _b_update           = torch.zeros_like(_b_expand).to(self.device)

        diffuseList = []
        iteration = [8, 16, 24, 32]

        for i in range(33):

            _f_expand_up    = torch.roll(_f_expand, -1, 2)
            _f_expand_down  = torch.roll(_f_expand, +1, 2)
            _f_expand_left  = torch.roll(_f_expand, -1, 3)
            _f_expand_right = torch.roll(_f_expand, +1, 3)

            _b_expand_up    = torch.roll(_b_expand, -1, 2)
            _b_expand_down  = torch.roll(_b_expand, +1, 2)
            _b_expand_left  = torch.roll(_b_expand, -1, 3)
            _b_expand_right = torch.roll(_b_expand, +1, 3)

            _f_laplace  = torch.mul(_f_expand_up, 1-_X_f_border_up) + torch.mul(_f_expand, _X_f_border_up) \
                        + torch.mul(_f_expand_down, 1-_X_f_border_down) + torch.mul(_f_expand, _X_f_border_down) \
                        + torch.mul(_f_expand_left, 1-_X_f_border_left) + torch.mul(_f_expand, _X_f_border_left) \
                        + torch.mul(_f_expand_right, 1-_X_f_border_right) + torch.mul(_f_expand, _X_f_border_right) \
                        - 4 * _f_expand 

            _b_laplace  = torch.mul(_b_expand_up, 1-_X_b_border_up) + torch.mul(_b_expand, _X_b_border_up) \
                        + torch.mul(_b_expand_down, 1-_X_b_border_down) + torch.mul(_b_expand, _X_b_border_down) \
                        + torch.mul(_b_expand_left, 1-_X_b_border_left) + torch.mul(_b_expand, _X_b_border_left) \
                        + torch.mul(_b_expand_right, 1-_X_b_border_right) + torch.mul(_b_expand, _X_b_border_right) \
                        - 4 * _b_expand 

            _f_update   = _f_expand + self.alpha * self.dt * _f_laplace
            _b_update   = _b_expand + self.alpha * self.dt * _b_laplace

            _f_diff     = torch.sum(torch.abs(torch.flatten(_f_update) - torch.flatten(_f_expand)))
            _b_diff     = torch.sum(torch.abs(torch.flatten(_b_update) - torch.flatten(_b_expand)))
            
            _f_expand   = _f_update
            _b_expand   = _b_update
        
            # print("[{0:3d}] err(f) : {1}, err(b) : {2}".format(i, _f_diff, _b_diff))
            
            _f = _f_expand[:,:,1:-1,1:-1]
            _b = _b_expand[:,:,1:-1,1:-1]
            
            _f = torch.mul(_f, X)
            _b = torch.mul(_b, 1-X)

            if i in iteration:
                diffuseList.append(_f)

        return diffuseList


def laplace_diffuse(I, X, num_iter=1, dt=0.25, alpha=1, device='cuda'):

    number_image_I  = I.shape[0]
    channel_image_I = I.shape[1]
    height_image_I  = I.shape[2]
    width_image_I   = I.shape[3]

    number_image_X  = X.shape[0]
    channel_image_X = X.shape[1]
    height_image_X  = X.shape[2]
    width_image_X   = X.shape[3]

    assert I.ndim == 4,         'dimension of input image I should be 4'
    assert X.ndim == 4,         'dimension of input mask X should be 4'
    assert I.ndim == X.ndim,    'dimension of input image I should match dimension of input mask X'

    p2d = (1, 1, 1, 1)

    _I_expand           = F.pad(I, p2d, 'replicate')
    _X_f_expand         = F.pad(X, p2d, 'replicate') 
    _X_b_expand         = F.pad(1-X, p2d, 'replicate')

    _X_f_expand_up      = torch.roll(_X_f_expand, -1, 2)
    _X_f_expand_down    = torch.roll(_X_f_expand, +1, 2)
    _X_f_expand_left    = torch.roll(_X_f_expand, -1, 3)
    _X_f_expand_right   = torch.roll(_X_f_expand, +1, 3)

    _X_b_expand_up      = torch.roll(_X_b_expand, -1, 2)
    _X_b_expand_down    = torch.roll(_X_b_expand, +1, 2)
    _X_b_expand_left    = torch.roll(_X_b_expand, -1, 3)
    _X_b_expand_right   = torch.roll(_X_b_expand, +1, 3)

    _X_f_border_up      = torch.mul(_X_f_expand, 1-_X_f_expand_up)
    _X_f_border_down    = torch.mul(_X_f_expand, 1-_X_f_expand_down)
    _X_f_border_left    = torch.mul(_X_f_expand, 1-_X_f_expand_left)
    _X_f_border_right   = torch.mul(_X_f_expand, 1-_X_f_expand_right)

    _X_b_border_up      = torch.mul(_X_b_expand, 1-_X_b_expand_up)
    _X_b_border_down    = torch.mul(_X_b_expand, 1-_X_b_expand_down)
    _X_b_border_left    = torch.mul(_X_b_expand, 1-_X_b_expand_left)
    _X_b_border_right   = torch.mul(_X_b_expand, 1-_X_b_expand_right)

    _f_expand           = torch.mul(_I_expand, _X_f_expand)
    _b_expand           = torch.mul(_I_expand, _X_b_expand)
    _f_update           = torch.zeros_like(_f_expand).to(device)
    _b_update           = torch.zeros_like(_b_expand).to(device)


    for i in range(num_iter):

        _f_expand_up    = torch.roll(_f_expand, -1, 2)
        _f_expand_down  = torch.roll(_f_expand, +1, 2)
        _f_expand_left  = torch.roll(_f_expand, -1, 3)
        _f_expand_right = torch.roll(_f_expand, +1, 3)

        _b_expand_up    = torch.roll(_b_expand, -1, 2)
        _b_expand_down  = torch.roll(_b_expand, +1, 2)
        _b_expand_left  = torch.roll(_b_expand, -1, 3)
        _b_expand_right = torch.roll(_b_expand, +1, 3)

        _f_laplace  = torch.mul(_f_expand_up, 1-_X_f_border_up) + torch.mul(_f_expand, _X_f_border_up) \
                    + torch.mul(_f_expand_down, 1-_X_f_border_down) + torch.mul(_f_expand, _X_f_border_down) \
                    + torch.mul(_f_expand_left, 1-_X_f_border_left) + torch.mul(_f_expand, _X_f_border_left) \
                    + torch.mul(_f_expand_right, 1-_X_f_border_right) + torch.mul(_f_expand, _X_f_border_right) \
                    - 4 * _f_expand 

        _b_laplace  = torch.mul(_b_expand_up, 1-_X_b_border_up) + torch.mul(_b_expand, _X_b_border_up) \
                    + torch.mul(_b_expand_down, 1-_X_b_border_down) + torch.mul(_b_expand, _X_b_border_down) \
                    + torch.mul(_b_expand_left, 1-_X_b_border_left) + torch.mul(_b_expand, _X_b_border_left) \
                    + torch.mul(_b_expand_right, 1-_X_b_border_right) + torch.mul(_b_expand, _X_b_border_right) \
                    - 4 * _b_expand 

        _f_update   = _f_expand + alpha * dt * _f_laplace
        _b_update   = _b_expand + alpha * dt * _b_laplace

        _f_diff     = torch.sum(torch.abs(torch.flatten(_f_update) - torch.flatten(_f_expand)))
        _b_diff     = torch.sum(torch.abs(torch.flatten(_b_update) - torch.flatten(_b_expand)))

        _f_expand   = _f_update
        _b_expand   = _b_update

        print("[{0:3d}] err(f) : {1}, err(b) : {2}".format(i, _f_diff, _b_diff))

    _f = _f_expand[:,:,1:-1,1:-1]
    _b = _b_expand[:,:,1:-1,1:-1]

    _f = torch.mul(_f, X)
    _b = torch.mul(_b, 1-X)

    return (_f, _b)

"test the lapalce function"
if __name__ == "__main__":
    def convert_image_np(inp, image):
        """Convert a Tensor to numpy image."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array((0.485, 0.456, 0.406))
        std = np.array((0.229, 0.224, 0.225))
        if image:
            inp = std * inp + mean
            inp = np.clip(inp, 0, 1)

        return inp

    X = torch.zeros((1,1,224,224))
    X[:,:,40:120,40:120] = 1

    transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
        ])
    trainset = torchvision.datasets.STL10(root='../../../../dataset/stl10', split='test', download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=1)
    for batch, (data, target) in enumerate(train_loader):

        f1, b1 = laplace_diffuse(data, X, num_iter=1)
        f2, b2 = laplace_diffuse(data, X, num_iter=8)
        f3, b3 = laplace_diffuse(data, X, num_iter=16)
        f4, b4 = laplace_diffuse(data, X, num_iter=32)
        f5, b5 = laplace_diffuse(data, X, num_iter=64)
        f6, b6 = laplace_diffuse(data, X, num_iter=128)
        f7, b7 = laplace_diffuse(data, X, num_iter=256)
        f8, b8 = laplace_diffuse(data, X, num_iter=512)
        f1 = f1[0].numpy().transpose((1, 2, 0))
        f2 = f2[0].numpy().transpose((1, 2, 0))
        f3 = f3[0].numpy().transpose((1, 2, 0))
        f4 = f4[0].numpy().transpose((1, 2, 0))
        f5 = f5[0].numpy().transpose((1, 2, 0))
        f6 = f6[0].numpy().transpose((1, 2, 0))
        f7 = f7[0].numpy().transpose((1, 2, 0))
        f8 = f8[0].numpy().transpose((1, 2, 0))
        b1 = b1[0].numpy().transpose((1, 2, 0))
        b2 = b2[0].numpy().transpose((1, 2, 0))
        b3 = b3[0].numpy().transpose((1, 2, 0))
        b4 = b4[0].numpy().transpose((1, 2, 0))
        b5 = b5[0].numpy().transpose((1, 2, 0))
        b6 = b6[0].numpy().transpose((1, 2, 0))
        b7 = b7[0].numpy().transpose((1, 2, 0))
        b8 = b8[0].numpy().transpose((1, 2, 0))

        plt.subplot(8,2,1)
        plt.imshow(f1)
        plt.subplot(8,2,2)
        plt.imshow(b1)
        plt.subplot(8,2,3)
        plt.imshow(f2)
        plt.subplot(8,2,4)
        plt.imshow(b2)
        plt.subplot(8,2,5)
        plt.imshow(f3)
        plt.subplot(8,2,6)
        plt.imshow(b3)
        plt.subplot(8,2,7)
        plt.imshow(f4)
        plt.subplot(8,2,8)
        plt.imshow(b4)
        plt.subplot(8,2,9)
        plt.imshow(f5)
        plt.subplot(8,2,10)
        plt.imshow(b5)
        plt.subplot(8,2,11)
        plt.imshow(f6)
        plt.subplot(8,2,12)
        plt.imshow(b6)
        plt.subplot(8,2,13)
        plt.imshow(f7)
        plt.subplot(8,2,14)
        plt.imshow(b7)
        plt.subplot(8,2,15)
        plt.imshow(f8)
        plt.subplot(8,2,16)
        plt.imshow(b8)

        # plt.tight_layout()
       

        # in_grid = convert_image_np(
        #         torchvision.utils.make_grid(img, nrow=4), False)
        # diffused_grid = convert_image_np(
        #     torchvision.utils.make_grid(blur, nrow=4), False)

        plt.savefig('../test.png')

        exit(1)