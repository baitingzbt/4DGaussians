#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np

@torch.no_grad()
def get_mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

@torch.no_grad()
def psnr(img1, img2, mask=None):
    if mask is not None:
        img1 = img1.flatten(1)
        img2 = img2.flatten(1)
        mask = mask.flatten(1).repeat(3,1)
        mask = torch.where(mask != 0, True, False)
        img1 = img1[mask]
        img2 = img2[mask]
    mse = get_mse(img1, img2)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse.float()))
    if mask is not None:
        if torch.isinf(psnr).any():
            print(mse.mean(),psnr.mean())
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse.float()))
            psnr = psnr[~torch.isinf(psnr)]
        
    return psnr

# no batch nor batch-dim here, single images only
def psnr_np(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    psnr = 20 * np.log10(1.0 / np.sqrt(mse))
    return psnr