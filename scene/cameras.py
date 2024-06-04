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
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

# class Camera():
#     def __init__(
#         self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
#         image_name, uid,
#         trans=np.array([0.0, 0.0, 0.0]),
#         scale=1.0,
#         data_device = "cuda",
#         time = 0,
#         mask = None,
#         depth = None,
#         force: np.ndarray = None,
#         force_idx: int = -1
#     ) -> None:

#         self.uid = uid
#         self.colmap_id = colmap_id
#         self.R = R
#         self.T = T
#         self.FoVx = FoVx
#         self.FoVy = FoVy
#         self.image_name = image_name
#         self.time = time
#         self.data_device = torch.device(data_device)
#         # self.prev_image = prev_image.clamp(0.0, 1.0)[:3, :, :]
#         self.original_image = image.clamp(0.0, 1.0)[:3, :, :]
#         self.image_width = self.original_image.shape[2]
#         self.image_height = self.original_image.shape[1]
#         if gt_alpha_mask is not None:
#             self.original_image *= gt_alpha_mask
#         else:
#             self.original_image *= torch.ones((1, self.image_height, self.image_width))
#         self.depth = depth
#         self.mask = mask
#         self.force = force
#         self.force_idx = force_idx
#         self.zfar = 100.0
#         self.znear = 0.01
#         self.trans = trans
#         self.scale = scale
#         self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale), dtype=torch.float32).transpose(0, 1)
#         self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1)
#         self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
#         self.camera_center = self.world_view_transform.inverse()[3, :3]

class Camera():
    def __init__(
        self,
        R,
        T,
        FoVx,
        FoVy,
        image,
        depth,
        mask,
        prev_frames,
        next_frames,
        time = 0,
        frame_step = 0,
        force: np.ndarray = None,
        full_force: np.ndarray = None,
        force_idx: int = -1,
        pose_idx: int = -1,
        pos_idx: int = -1,
        unit_time: float = 1.,
    ) -> None:
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.time = time
        self.frame_step = frame_step
        self.data_device = torch.device("cuda")
        # 
        self.original_image = image.clamp(0.0, 1.0)[:3, :, :]
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]
        self.depth = depth
        self.prev_frames = None if prev_frames is None else prev_frames.clamp(0.0, 1.0)
        self.next_frames = None if next_frames is None else next_frames.clamp(0.0, 1.0)
        ''' start, for idx0based prev frame '''
        # self.prev_frames = torch.ones_like(self.prev_frames) * pos_idx / 720
        ''' end, 720 is a magic number '''
        self.mask = None if mask is None else mask[0]
        self.force = force
        self.full_force = full_force
        self.pose_idx = pose_idx
        self.force_idx = force_idx
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, np.array([0.0, 0.0, 0.0]), 1.), dtype=torch.float32).transpose(0, 1)
        projection_matrix = getProjectionMatrix(znear=0.01, zfar=100., fovX=self.FoVx, fovY=self.FoVy).transpose(0,1) # 
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.pos_idx = pos_idx
        self.prev_state = None # means3D
        self.unit_time = unit_time