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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from utils.sh_utils import eval_sh
import numpy as np
from copy import deepcopy
from utils.general_utils import knn
from scene.dataset_readers import START_FRAME, MAX_FRAME
from typing import List
from collections import defaultdict

# ANCHORS = np.array([
#     0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0
# ])
ANCHORS = np.array([
    0, 0.5, 1.0
])
# [0-0.25, 0.25-0.75, 0.75-1.0]
# [0,      0.5,        1.0]

# [0/40,  1/40, ..., 39/40]
def cast_time_to_anchor(t: float) -> float:
    return float(ANCHORS[np.argmin(np.abs(ANCHORS - t))])


def render_for_state(
    cam: Camera,
    pc: GaussianModel,
):
    # Set up rasterization configuration
    n_points = pc.get_xyz.shape[0]
    device = pc.get_xyz.device
    time = torch.tensor(cam.time).to(dtype=torch.float32, device=device).repeat(n_points, 1)
    force = torch.tensor(cam.force).to(dtype=torch.float32, device=device).repeat(n_points, 1)
    if cam.prev_state is not None and pc._deformation.recur:
        means3D = cam.prev_state['means3D']
        opacity = cam.prev_state['opacity']
        shs = cam.prev_state['shs']
        scales = cam.prev_state['scales']
        rotations = cam.prev_state['rotations']
    else:
        opacity = pc._opacity
        shs = pc.get_features
        scales = pc._scaling
        rotations = pc._rotation
        means3D = pc.get_xyz
    means3D_final, scales_final, rotations_final, opacity_final, shs_final = pc._deformation.forward_dynamic(
        means3D, scales, rotations, opacity, shs, time, force, cam.prev_frames
    )

    state = {
        'means3D': means3D_final.detach(),
        'opacity': opacity_final.detach(),
        'shs': shs_final.detach(),
        'rotations': rotations_final.detach(),
        'scales': scales_final.detach()
    }
    return state

def render(
    cam: Camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier = 1.0,
    stage="fine"
):
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda")
    # if not ('train' in stage or 'test' in stage):
    screenspace_points.retain_grad()

    # Set up rasterization configuration
    # cam = cams[1]
    raster_settings = GaussianRasterizationSettings(
        image_height=int(cam.image_height),
        image_width=int(cam.image_width),
        tanfovx=math.tan(cam.FoVx * 0.5),
        tanfovy=math.tan(cam.FoVy * 0.5),
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=cam.world_view_transform.cuda(),
        projmatrix=cam.full_proj_transform.cuda(),
        sh_degree=pc.active_sh_degree,
        campos=cam.camera_center.cuda(),
        prefiltered=False,
        debug=pipe.debug
    )
    n_points = pc.get_xyz.shape[0]
    device = pc.get_xyz.device

    # if "anchor" in stage:
    #     force *= 0.0
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    means2D = screenspace_points
    opacity = pc._opacity
    shs = pc.get_features
    scales = pc._scaling
    rotations = pc._rotation
    means3D = pc.get_xyz
    if "coarse" in stage:
        means3D_final, scales_final, rotations_final, opacity_final, shs_final \
            = means3D, scales, rotations, opacity, shs
    else:
        _time = cam.time
        _force = cam.force
        if "anchor" in stage:
            _time = cast_time_to_anchor(_time)
        time = torch.tensor(_time).to(dtype=torch.float32, device=device).repeat(n_points, 1)
        time_prev = torch.ones_like(time) * (_time - 1 / (MAX_FRAME - 1))
        time_nxt = torch.ones_like(time) * (_time + 1 / (MAX_FRAME - 1))
        force = torch.tensor(_force).to(dtype=torch.float32, device=device).repeat(n_points, 1)
        # if cam.prev_state is not None and pc._deformation.recur:
        #     means3D = cam.prev_state['means3D']  # (n_pts, 3)
        #     opacity = cam.prev_state['opacity']  # (n_pts, 3)
        #     shs = cam.prev_state['shs']  # (n_pts, 4)
        #     scales = cam.prev_state['scales']  # (n_pts, 3)
        #     rotations = cam.prev_state['rotations']  # (n_pts, 4)
        prev_frames = cam.prev_frames
        means3D_final, scales_final, rotations_final, opacity_final, shs_final \
            = pc._deformation.forward_dynamic(means3D, scales, rotations, opacity, shs, time, force, prev_frames)
    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    opacity_final = pc.opacity_activation(opacity_final)
    # if means3D_final.abs().max().detach().item() > 100:
    #     print(f"num of irregular pts: {(means3D_final.abs() > 100).sum()}")
    
    reg_dict = defaultdict(lambda: torch.tensor(0))
    if ("coarse" in stage) or ("anchor" in stage):
        pass
    # elif pc._deformation.recur:
    #     velocity = means3D_final - means3D
    #     k = 10
    #     idx, dist = knn(means3D_final[None].contiguous().detach(), means3D_final[None].contiguous().detach(), k)
    #     vel_dist = torch.norm(velocity[idx] - velocity[None, :, None], p=2, dim=-1)
    #     weight = torch.exp(-100 * dist) # * (vel_dist < 0.25).float()  |  farther the 3d distance, lower the impact?
    #     reg_dict['knn'] = (weight * vel_dist).sum() / k / n_points
    #     reg_dict['momentum'] = velocity.abs().mean()
    else:
        prev_frames_prev = None # cam[0].prev_frames
        prev_frames_nxt = None # cam[2].prev_frames
        means3D_prev, scales_prev, rotations_prev, opacity_prev, shs_prev \
            = pc._deformation.forward_dynamic(means3D, scales, rotations, opacity, shs, time_prev, force, prev_frames_prev)
        means3D_nxt, scales_nxt, rotations_next, opacity_next, shs_nxt \
            = pc._deformation.forward_dynamic(means3D, scales, rotations, opacity, shs, time_nxt, force, prev_frames_nxt)
        
        scales_prev = pc.scaling_activation(scales_prev)
        scales_nxt = pc.scaling_activation(scales_nxt)        
        momentum_all = torch.abs(means3D_nxt + means3D_prev - 2 * means3D_final) # |pos_prev - pos_cur| + |pos_cur - pos_next|
        reg_dict['momentum'] = momentum_all.mean()
        reg_dict['scales'] = torch.abs(scales_nxt + scales_prev - 2 * scales_final).sum(dim=1).mean()
        reg_dict['shs'] = torch.abs(shs_nxt + shs_prev - 2 * shs_final).mean()
        # fastest = torch.topk(momentum_all, k=500, largest=True, dim=0).indices.detach().cpu()
        ############# KNN rigid - nearby points have similar velo ###############
        velocity = (means3D_nxt - means3D_prev) / 2 # approxiamtion
        k = min(100, n_points)
        idx, dist = knn(means3D_final[None].contiguous().detach(), means3D_final[None].contiguous().detach(), k)
        vel_dist = torch.norm(velocity[idx] - velocity[None, :, None], p=2, dim=-1)
        weight = torch.exp(-100 * dist)
        # some points teleports when deforming, use this threshold to not learn to teleport with neighbors
        reg_dict['knn'] = (weight * vel_dist).sum() / k / n_points

    rendered_image, radii, depth = rasterizer.forward(
        means3D = means3D_final,
        means2D = means2D,
        shs = shs_final,
        colors_precomp = None,
        opacities = opacity_final,
        scales = scales_final,
        rotations = rotations_final,
        cov3D_precomp = None
    )
    if "coarse" not in stage and cam.next_frames is not None:
        next_renders = render_next_frames(cam, pc, rasterizer)
    else:
        next_renders = None
    return rendered_image, screenspace_points, radii > 0, radii, depth, reg_dict, next_renders

def render_next_frames(
    cam: Camera,
    pc: GaussianModel,
    rasterizer: GaussianRasterizer
) -> torch.TensorType:
    n = cam.next_frames.shape[0]
    out = torch.zeros_like(cam.next_frames)
    n_points = pc.get_xyz.shape[0]
    device = pc.get_xyz.device
    for i in range(1, n+1):
        if i + cam.frame_step >= MAX_FRAME:
            continue  # next frame and pred both kept as 0, no loss incurred
        t = cam.time + i * cam.unit_time
        time = torch.tensor(t).to(dtype=torch.float32, device=device).repeat(n_points, 1)
        force = torch.tensor(cam.force).to(dtype=torch.float32, device=device).repeat(n_points, 1)
        screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda")
        screenspace_points.retain_grad()
        means2D = screenspace_points
        opacity = pc._opacity
        shs = pc.get_features
        scales = pc._scaling
        rotations = pc._rotation
        means3D = pc.get_xyz
        means3D_final, scales_final, rotations_final, opacity_final, shs_final \
            = pc._deformation.forward_dynamic(means3D, scales, rotations, opacity, shs, time, force, cam.prev_frames)
        scales_final = pc.scaling_activation(scales_final)
        rotations_final = pc.rotation_activation(rotations_final)
        opacity_final = pc.opacity_activation(opacity_final)
        out[i-1] = rasterizer.forward(
            means3D = means3D_final,
            means2D = means2D,
            shs = shs_final,
            colors_precomp = None,
            opacities = opacity_final,
            scales = scales_final,
            rotations = rotations_final,
            cov3D_precomp = None
        )[0]
    return out

@torch.no_grad()
def render_for_image(
    viewpoint_camera: Camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier = 1.0,
    stage="fine"
):
    # Set up rasterization configuration
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=math.tan(viewpoint_camera.FoVx * 0.5),
        tanfovy=math.tan(viewpoint_camera.FoVy * 0.5),
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=False
    )
    n_points = pc.get_xyz.shape[0]
    device = pc.get_xyz.device
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    means2D = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda")

    if (viewpoint_camera.prev_state is not None) and pc._deformation.recur and ('coarse' not in stage):
        means3D = viewpoint_camera.prev_state['means3D']
        opacity = viewpoint_camera.prev_state['opacity']
        shs = viewpoint_camera.prev_state['shs']
        scales = viewpoint_camera.prev_state['scales']
        rotations = viewpoint_camera.prev_state['rotations']
    else:
        means3D = pc.get_xyz
        opacity = pc._opacity
        shs = pc.get_features
        scales = pc._scaling
        rotations = pc._rotation

    if "coarse" in stage:
        means3D_final, scales_final, rotations_final, opacity_final, shs_final \
            = means3D, scales, rotations, opacity, shs
    else:
        _time = viewpoint_camera.time
        _force = viewpoint_camera.force
        # if "anchor" in stage:
        #     _time = cast_time_to_anchor(_time)
        time = torch.tensor(_time).to(dtype=torch.float32, device=device).repeat(n_points, 1)
        force = torch.tensor(_force).to(dtype=torch.float32, device=device).repeat(n_points, 1)
        prev_frames = viewpoint_camera.prev_frames
        means3D_final, scales_final, rotations_final, opacity_final, shs_final \
            = pc._deformation.forward_dynamic(means3D, scales, rotations, opacity, shs, time, force, prev_frames)
    
    
    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    opacity_final = pc.opacity_activation(opacity_final)

    rendered_image, radii, depth = rasterizer.forward(
        means3D = means3D_final,
        means2D = means2D,
        shs = shs_final,
        colors_precomp = None,
        opacities = opacity_final,
        scales = scales_final,
        rotations = rotations_final,
        cov3D_precomp = None
    )
    return rendered_image
