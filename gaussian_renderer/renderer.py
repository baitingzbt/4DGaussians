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
    viewpoint_camera: Camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier = 1.0,
    override_color = None,
    stage="fine",
    training = True
):
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

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
        means3D = pc.get_xyz
        means3D_final, scales_final, rotations_final, opacity_final, shs_final \
            = means3D, scales, rotations, opacity, shs
    else:
        _time = viewpoint_camera.time
        _force = viewpoint_camera.force
        if "anchor" in stage:
            _time = cast_time_to_anchor(_time)
        time = torch.tensor(_time).to(dtype=torch.float32, device=device).repeat(n_points, 1)
        time_prev = torch.ones_like(time) * (_time - 1/39)
        time_nxt = torch.ones_like(time) * (_time + 1/39)
        force = torch.tensor(_force).to(dtype=torch.float32, device=device).repeat(n_points, 1)
        if viewpoint_camera.prev_state is not None and pc._deformation.recur_state:
            means3D = viewpoint_camera.prev_state['means3D']
            opacity = viewpoint_camera.prev_state['opacity']
            shs = viewpoint_camera.prev_state['shs']
            scales = viewpoint_camera.prev_state['scales']
            rotations = viewpoint_camera.prev_state['rotations']
        if viewpoint_camera.prev_hidden is not None and pc._deformation.recur_hidden:
            hidden = viewpoint_camera.prev_hidden.detach()
        means3D_final, scales_final, rotations_final, opacity_final, shs_final, _ \
            = pc._deformation.forward_dynamic(means3D, scales, rotations, opacity, shs, time, force, hidden)

    
    # recursion doesn't use momentum loss
    if ("coarse" in stage) or ("anchor" in stage) or pc._deformation.recur or not training:
        momentum_reg = torch.tensor(0.0)
        opacity_reg = torch.tensor(0.0)
        knn_reg = torch.tensor(0.0)
    else:
        # use only means now
        means3D_prev, _, _, opa_prev, _, _ \
            = pc._deformation.forward_dynamic(means3D, scales, rotations, opacity, shs, time_prev, force, None)
        means3D_nxt, _, _, opa_nxt, _, _ \
            = pc._deformation.forward_dynamic(means3D, scales, rotations, opacity, shs, time_nxt, force, None)
        momentum_reg = torch.abs(means3D_nxt + means3D_prev - 2 * means3D_final).mean()
        opacity_reg = torch.abs(opa_nxt + opa_prev - 2 * opacity_final).mean()
        ############# KNN rigid - nearby points have similar velo ###############
        velocity = (means3D_nxt - means3D_prev) / 2 # approxiamtion
        # opa_diff = abs(opacity_final - opa_prev) + abs(opacity_final - opa_nxt)
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
        # print((vel_dist>=0.2).sum(), vel_dist.max())
        # breakpoint()

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

    # # Rasterize visible Gaussians to image, obtain their radii (on screen).
    # indices = torch.tensor([  891, 22718,  7681, 21858, 20494,  1295, 18694, 15889, 18652,  6032,
    #     12477,  1329,  7347,  4525,   943,  1234,  1889, 27604,  5705, 29562,
    #     14005,  3003,  7483, 22540, 22390, 21726, 27588, 26637,  8312,  4710,
    #     13667,  4529, 18713, 30228, 11168,  5341, 17192, 21090, 30273, 26777,
    #      4680, 28817, 16273,  4248, 10958,  5673, 21867, 17663, 21549, 14384,
    #     12532, 13212, 23621, 27318, 17477, 17658, 23593, 14054,  2672, 21205,
    #     21111, 10757,   905,  2623,  2178, 23688,  5706, 28390, 28654, 10864,
    #     29727, 26100, 21251, 24680, 14894, 17079, 16851,  8432,  5843, 14779,
    #     21724,  4742, 22206, 25738, 18369,  6850,  3569,   332, 20809,  1712,
    #      4044,  1202,  1250, 14714, 21438, 25480,  7726, 18490, 24109,  1186],
    #    device='cuda:0')
    # rendered_image, radii, depth = rasterizer.forward(
    #     means3D = means3D_final[indices],
    #     means2D = means2D[indices],
    #     shs = shs_final[indices],
    #     colors_precomp = colors_precomp,
    #     opacities = opacity[indices],
    #     scales = scales_final[indices],
    #     rotations = rotations_final[indices],
    #     cov3D_precomp = None
    # )
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

