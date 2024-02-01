import functools
import math
import os
import time
from tkinter import W
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from utils.graphics_utils import apply_rotation, batch_quaternion_multiply
from scene.hexplane import HexPlaneField
from scene.grid import DenseGrid

class Deformation(nn.Module):
    def __init__(self, D=8, W=256, input_ch=27, input_ch_time=9, grid_pe=0, skips=[], args=None):
        super(Deformation, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_time = input_ch_time
        self.skips = skips
        self.grid_pe = grid_pe
        self.no_grid = args.no_grid
        self.grid = HexPlaneField(args.bounds, args.kplanes_config, args.multires)
        self.blend_time_force = args.blend_time_force
        self.args = args
        # self.args.empty_voxel=True
        if self.args.empty_voxel:
            self.empty_voxel = DenseGrid(channels=1, world_size=[64, 64, 64])
        if self.args.static_mlp:
            self.static_mlp = nn.Sequential(
                nn.ReLU(), nn.Linear(self.W, self.W, dtype=torch.float32),
                nn.ReLU(), nn.Linear(self.W, 1, dtype=torch.float32)
            )
        self.ratio=0
        self.create_net()

    @property
    def get_aabb(self):
        return self.grid.get_aabb

    def set_aabb(self, xyz_max, xyz_min):
        # print("Deformation Net Set aabb",xyz_max, xyz_min)
        self.grid.set_aabb(xyz_max, xyz_min)
        if self.args.empty_voxel:
            self.empty_voxel.set_aabb(xyz_max, xyz_min)
    
    def create_net(self):
        mlp_out_dim = 0
        if self.grid_pe != 0:
            grid_out_dim = self.grid.feat_dim + (self.grid.feat_dim) * 2 
        else:
            grid_out_dim = self.grid.feat_dim

        self.feature_out = [nn.Linear(mlp_out_dim + grid_out_dim, self.W, dtype=torch.float32, bias=True)]
        for i in range(self.D - 1):
            self.feature_out.append(nn.ReLU())
            self.feature_out.append(nn.Linear(self.W, self.W, dtype=torch.float32, bias=True))
        self.feature_out = nn.Sequential(*self.feature_out)

        self.pos_deform = nn.Sequential(
            nn.ReLU(), nn.Linear(self.W, self.W, dtype=torch.float32, bias=True),
            nn.ReLU(), nn.Linear(self.W, 3, dtype=torch.float32, bias=True)
        )
        self.scales_deform = nn.Sequential(
            nn.ReLU(), nn.Linear(self.W, self.W, dtype=torch.float32, bias=True),
            nn.ReLU(), nn.Linear(self.W, 3, dtype=torch.float32, bias=True)
        )
        self.rotations_deform = nn.Sequential(
            nn.ReLU(), nn.Linear(self.W, self.W, dtype=torch.float32, bias=True),
            nn.ReLU(), nn.Linear(self.W, 4, dtype=torch.float32, bias=True)
        )
        self.opacity_deform = nn.Sequential(
            nn.ReLU(), nn.Linear(self.W, self.W, dtype=torch.float32, bias=True),
            nn.ReLU(), nn.Linear(self.W, 1, dtype=torch.float32, bias=True)
        )
        self.force_embed = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 1))
        if self.blend_time_force:
            self.force_time_embed = nn.Sequential(nn.ReLU(), nn.Linear(2, 1))
        # self.force_deform = nn.Sequential(
        #     nn.ReLU(), nn.Linear(self.W, self.W, dtype=torch.float32, bias=True),
        #     nn.ReLU(), nn.Linear(self.W, 4, dtype=torch.float32, bias=True)
        # )
        self.shs_deform = nn.Sequential(
            nn.ReLU(), nn.Linear(self.W, self.W, dtype=torch.float32, bias=True),
            nn.ReLU(), nn.Linear(self.W, 16 * 3, dtype=torch.float32, bias=True)
        )

    def query_time_force(self, rays_pts_emb, scales_emb, rotations_emb, time_feature, time_emb, force_emb, prev_hidden):
        ''' embedding force from dim4 to dim1, to reduce numerical instability '''
        if force_emb is not None: # NOTE: if use_force = True
            force_emb = torch.exp(self.force_embed(force_emb / 10))
        if self.blend_time_force:
            time_emb = self.force_time_embed(torch.cat((time_emb, force_emb), dim=1))
            force_emb = None # NOTE: force information merged into time

        grid_feature = self.grid(rays_pts_emb[:, :3], time_emb, force_emb, prev_hidden)
        if self.grid_pe > 1:
            grid_feature = poc_fre(grid_feature, self.grid_pe)
        hidden = torch.cat([grid_feature], -1).to(dtype=torch.float32)
        hidden2 = self.feature_out(hidden)
        return hidden2


    @property
    def get_empty_ratio(self):
        return self.ratio
    
    def forward(
        self, rays_pts_emb, scales_emb=None, rotations_emb=None, opacity = None, shs_emb=None, 
        time_feature=None, time_emb=None, force_emb=None, prev_hidden=None,
    ):
        # print(f"defomation.py line 124:\n\tforwarding: time_emb = {time_emb}, prev_hidden = {prev_hidden}", )
        if time_emb is None:
            return self.forward_static(rays_pts_emb[:, :3])
        else:
            return self.forward_dynamic(rays_pts_emb, scales_emb, rotations_emb, opacity, shs_emb, time_feature, time_emb, force_emb, prev_hidden)
    

    def forward_static(self, rays_pts_emb):
        grid_feature = self.grid(rays_pts_emb[:,:3])
        grid_feature.to(dtype=torch.float32)
        dx = self.static_mlp(grid_feature)
        return rays_pts_emb[:, :3] + dx


    # TODO: EDIT THIS FUNCTION OR WRITE A NEW ONE FOR FORCE
    def forward_dynamic(self, rays_pts_emb, scales_emb, rotations_emb, opacity_emb, shs_emb, time_feature, time_emb, force_emb, prev_hidden):
        time_input = time_emb[:, :1] if self.args.use_time else None
        force_input = force_emb[:, 3:7] if self.args.use_force else None # NOTE: drops xyz, keep dir+strength only
        hidden = self.query_time_force(rays_pts_emb, scales_emb, rotations_emb, time_feature, time_input, force_input, prev_hidden)

        if self.args.static_mlp:
            mask = self.static_mlp(hidden)
        elif self.args.empty_voxel:
            mask = self.empty_voxel(rays_pts_emb[:, :3])
        else: # NO MASK
            mask = torch.ones_like(opacity_emb[:, 0], dtype=torch.float32).unsqueeze(-1)

        if self.args.no_dx:
            pts = rays_pts_emb[:, :3]
        else:
            dx = self.pos_deform(hidden)
            pts = rays_pts_emb[:, :3] * mask + dx


        if self.args.no_ds:
            scales = scales_emb[:, :3]
        else:
            ds = self.scales_deform(hidden)
            scales = scales_emb[:, :3] * mask + ds
            
        if self.args.no_dr:
            rotations = rotations_emb[:, :4]
        else:
            dr = self.rotations_deform(hidden)
            if self.args.apply_rotation:
                rotations = batch_quaternion_multiply(rotations_emb, dr)
            else:
                rotations = rotations_emb[:, :4] + dr

        if self.args.no_do:
            opacity = opacity_emb[:, :1] 
        else:
            do = self.opacity_deform(hidden) 
            opacity = opacity_emb[:, :1] * mask + do

        if self.args.no_dshs:
            shs = shs_emb
        else:
            dshs = self.shs_deform(hidden).reshape([shs_emb.shape[0], 16, 3])
            shs = shs_emb * mask.unsqueeze(-1) + dshs
        # print(f"\thidden output = {hidden.shape} deformation.py line 176")
        return pts, scales, rotations, opacity, shs #, hidden
    
    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" not in name:
                parameter_list.append(param)
        return parameter_list
    def get_grid_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" in name:
                parameter_list.append(param)
        return parameter_list
    

# THE NETWORK THAT OUTPUTS RENDERED IMAGES
class deform_network(nn.Module):
    def __init__(self, args) :
        super(deform_network, self).__init__()
        net_width = args.net_width
        timebase_pe = args.timebase_pe
        defor_depth= args.defor_depth
        posbase_pe= args.posebase_pe
        scale_rotation_pe = args.scale_rotation_pe
        opacity_pe = args.opacity_pe
        timenet_width = args.timenet_width
        timenet_output = args.timenet_output
        grid_pe = args.grid_pe
        times_ch = 2*timebase_pe+1
        self.timenet = nn.Sequential(
            nn.Linear(times_ch, timenet_width),
            nn.ReLU(),
            nn.Linear(timenet_width, timenet_output)
        )
        self.deformation_net = Deformation(
            W=net_width,
            D=defor_depth,
            input_ch=(3)+(3*(posbase_pe))*2,
            grid_pe=grid_pe,
            input_ch_time=timenet_output,
            args=args
        )
        self.register_buffer('time_poc', torch.FloatTensor([(2**i) for i in range(timebase_pe)]))
        self.register_buffer('pos_poc', torch.FloatTensor([(2**i) for i in range(posbase_pe)]))
        self.register_buffer('rotation_scaling_poc', torch.FloatTensor([(2**i) for i in range(scale_rotation_pe)]))
        self.register_buffer('opacity_poc', torch.FloatTensor([(2**i) for i in range(opacity_pe)]))
        self.apply(initialize_weights)
        # print(self)

    def forward(self, point, scales=None, rotations=None, opacity=None, shs=None, times_sel=None, force=None, prev_hidden=None):
        return self.forward_dynamic(point, scales, rotations, opacity, shs, times_sel, force, prev_hidden)

    @property
    def get_aabb(self):
        return self.deformation_net.get_aabb
    
    @property
    def get_empty_ratio(self):
        return self.deformation_net.get_empty_ratio
        
    def forward_static(self, points):
        points = self.deformation_net(points)
        return points
    
    def forward_dynamic(self, point, scales=None, rotations=None, opacity=None, shs=None, times_sel=None, force=None, prev_hidden=None):
        point_emb = poc_fre(point, self.pos_poc)
        scales_emb = poc_fre(scales, self.rotation_scaling_poc)
        rotations_emb = poc_fre(rotations, self.rotation_scaling_poc)
        means3D, scales, rotations, opacity, shs = self.deformation_net.forward(
            point_emb, scales_emb, rotations_emb, opacity, shs, None, times_sel, force, prev_hidden
        )
        return means3D, scales, rotations, opacity, shs #, hidden
    
    def get_mlp_parameters(self):
        return self.deformation_net.get_mlp_parameters() + list(self.timenet.parameters())

    def get_grid_parameters(self):
        return self.deformation_net.get_grid_parameters()

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight,gain=1)
        if m.bias is not None:
            init.xavier_uniform_(m.weight,gain=1)

def poc_fre(input_data, poc_buf):
    input_data_emb = (input_data.unsqueeze(-1) * poc_buf).flatten(-2)
    input_data_sin = input_data_emb.sin()
    input_data_cos = input_data_emb.cos()
    input_data_emb = torch.cat([input_data, input_data_sin,input_data_cos], -1)
    return input_data_emb