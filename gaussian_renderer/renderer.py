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
    if cam.prev_state is not None and pc._deformation.recur_state:
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
        
    means3D_final, scales_final, rotations_final, opacity_final, shs_final, _ = pc._deformation.forward_dynamic(
        means3D, scales, rotations, opacity, shs, time, force, cam.prev_hidden
    )

    state = {
        'means3D': means3D_final.detach(),
        'opacity': opacity_final.detach(),
        'shs': shs_final.detach(),
        'rotations': rotations_final.detach(),
        'scales': scales_final.detach()
    }
    return state

def render_for_hidden(
    cam: Camera,
    pc: GaussianModel,
):
    # Set up rasterization configuration
    time = torch.tensor(cam.time).to(dtype=torch.float32, device=pc.get_xyz.device).repeat(pc.get_xyz.shape[0], 1)
    force = torch.tensor(cam.force).to(dtype=torch.float32, device=pc.get_xyz.device).repeat(pc.get_xyz.shape[0], 1)
    opacity = pc._opacity
    shs = pc.get_features
    scales = pc._scaling
    rotations = pc._rotation
    means3D = pc.get_xyz
    hidden = cam.prev_hidden
    # return the new hidden
    return pc._deformation.forward_dynamic(means3D, scales, rotations, opacity, shs, time, force, hidden)[-1].detach()

def render(
    viewpoint_cameras: List[Camera],
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier = 1.0,
    override_color = None,
    stage="fine"
):
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    viewpoint_camera = viewpoint_cameras[1]
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
    hidden = None
    if "coarse" in stage:
        means3D_final, scales_final, rotations_final, opacity_final, shs_final \
            = means3D, scales, rotations, opacity, shs
    else:
        _time = viewpoint_camera.time
        _force = viewpoint_camera.force
        if "anchor" in stage:
            _time = cast_time_to_anchor(_time)
        time = torch.tensor(_time).to(dtype=torch.float32, device=device).repeat(n_points, 1)
        time_prev = torch.ones_like(time) * (_time - 1 / (MAX_FRAME - START_FRAME))
        time_nxt = torch.ones_like(time) * (_time + 1 / (MAX_FRAME - START_FRAME))
        force = torch.tensor(_force).to(dtype=torch.float32, device=device).repeat(n_points, 1)
        if viewpoint_camera.prev_state is not None and pc._deformation.recur_state:
            means3D = viewpoint_camera.prev_state['means3D']  # (n_pts, 3)
            opacity = viewpoint_camera.prev_state['opacity']  # (n_pts, 3)
            shs = viewpoint_camera.prev_state['shs']  # (n_pts, 4)
            scales = viewpoint_camera.prev_state['scales']  # (n_pts, 3)
            rotations = viewpoint_camera.prev_state['rotations']  # (n_pts, 4)
        if viewpoint_camera.prev_hidden is not None and pc._deformation.recur_hidden:
            hidden = viewpoint_camera.prev_hidden.detach()
        prev_frames = viewpoint_camera.prev_frames
        means3D_final, scales_final, rotations_final, opacity_final, shs_final, _ \
            = pc._deformation.forward_dynamic(means3D, scales, rotations, opacity, shs, time, force, hidden, prev_frames)
    if means3D_final.abs().max().detach().item() > 100:
        print(f"num of irregular pts: {(means3D_final.abs() > 100).sum()}")
    
    # recursion doesn't use momentum loss
    if ("coarse" in stage) or ("anchor" in stage):
        momentum_reg = torch.tensor(0.0)
        opacity_reg = torch.tensor(0.0)
        knn_reg = torch.tensor(0.0)
    elif pc._deformation.recur:
        momentum_reg = torch.tensor(0.0)
        opacity_reg = torch.tensor(0.0)
        velocity = means3D_final - means3D
        k = 10
        idx, dist = knn(
            means3D_final[None].contiguous().detach(), 
            means3D_final[None].contiguous().detach(), 
            k
        )
        vel_dist = torch.norm(velocity[idx] - velocity[None, :, None], p=2, dim=-1)
        weight = torch.exp(-100 * dist) # * (vel_dist < 0.25).float()
        knn_reg = (weight * vel_dist).sum() / k / means3D_final.shape[0]
    else:
        prev_frames_prev = viewpoint_cameras[0].prev_frames
        prev_frames_nxt = viewpoint_cameras[2].prev_frames
        means3D_prev, _, _, opa_prev, _, _ \
            = pc._deformation.forward_dynamic(means3D, scales, rotations, opacity, shs, time_prev, force, None, prev_frames_prev)
        means3D_nxt, _, _, opa_nxt, _, _ \
            = pc._deformation.forward_dynamic(means3D, scales, rotations, opacity, shs, time_nxt, force, None, prev_frames_nxt)
        # breakpoint()
        momentum_reg = torch.abs(means3D_nxt + means3D_prev - 2 * means3D_final).mean()
        opacity_reg = torch.abs(opa_nxt + opa_prev - 2 * opacity_final).mean()
        ############# KNN rigid - nearby points have similar velo ###############
        velocity = (means3D_nxt - means3D_prev) / 2 # approxiamtion
        k = 10
        idx, dist = knn(
            means3D_final[None].contiguous().detach(), 
            means3D_final[None].contiguous().detach(), 
            k
        )
        
        vel_dist = torch.norm(velocity[idx] - velocity[None, :, None], p=2, dim=-1)
        weight = torch.exp(-100 * dist) # * (vel_dist < 0.25).float()
        # some points teleports when deforming, use this threshold to not learn to teleport
        # with neighbors
        knn_reg = (weight * vel_dist).sum() / k / means3D_final.shape[0]
        ############# KNN rigid - nearby points have similar velo ###############

    # print(f"scales final: {scales_final}")
    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    opacity = pc.opacity_activation(opacity_final)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.cuda().repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    else:
        colors_precomp = override_color

    rendered_image, radii, depth = rasterizer.forward(
        means3D = means3D_final,
        means2D = means2D,
        shs = shs_final,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales_final,
        rotations = rotations_final,
        cov3D_precomp = None
    )
    return rendered_image, screenspace_points, radii > 0, radii, depth, momentum_reg, opacity_reg, knn_reg


@torch.no_grad()
def render_for_image(
    viewpoint_camera: Camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier = 1.0,
    override_color = None,
    stage="fine"
):
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0

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
        debug=pipe.debug
    )
    n_points = pc.get_xyz.shape[0]
    device = pc.get_xyz.device
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    means2D = screenspace_points
    opacity = pc._opacity
    shs = pc.get_features
    scales = pc._scaling
    rotations = pc._rotation
    means3D = pc.get_xyz
    hidden = None
    if "coarse" in stage:
        means3D_final, scales_final, rotations_final, opacity_final, shs_final \
            = means3D, scales, rotations, opacity, shs
    else:
        _time = viewpoint_camera.time
        _force = viewpoint_camera.force
        if "anchor" in stage:
            _time = cast_time_to_anchor(_time)
        time = torch.tensor(_time).to(dtype=torch.float32, device=device).repeat(n_points, 1)
        force = torch.tensor(_force).to(dtype=torch.float32, device=device).repeat(n_points, 1)
        if viewpoint_camera.prev_state is not None and pc._deformation.recur_state:
            means3D = viewpoint_camera.prev_state['means3D']
            opacity = viewpoint_camera.prev_state['opacity']
            shs = viewpoint_camera.prev_state['shs']
            scales = viewpoint_camera.prev_state['scales']
            rotations = viewpoint_camera.prev_state['rotations']
        if viewpoint_camera.prev_hidden is not None and pc._deformation.recur_hidden:
            hidden = viewpoint_camera.prev_hidden.detach()
        prev_frames = viewpoint_camera.prev_frames
        means3D_final, scales_final, rotations_final, opacity_final, shs_final, _ \
            = pc._deformation.forward_dynamic(means3D, scales, rotations, opacity, shs, time, force, hidden, prev_frames)
    if means3D_final.abs().max().detach().item() > 100:
        print(f"num of irregular pts: {(means3D_final.abs() > 100).sum()}")
        breakpoint()
    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    opacity = pc.opacity_activation(opacity_final)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.cuda().repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    else:
        colors_precomp = override_color
    rendered_image, radii, depth = rasterizer.forward(
        means3D = means3D_final,
        means2D = means2D,
        shs = shs_final,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales_final,
        rotations = rotations_final,
        cov3D_precomp = None
    )
    return rendered_image
