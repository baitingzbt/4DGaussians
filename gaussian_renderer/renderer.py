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

def get_pos_t0(pc: GaussianModel):
    means3D = pc.get_xyz
    scales = pc._scaling
    rotations = pc._rotation
    opacity = pc._opacity
    time = torch.tensor(0.0).to(means3D.device).repeat(means3D.shape[0], 1)
    # pc._deformation.forward(means3D, scales, rotations, opacity, shs, time, force)
    deformation_point = pc._deformation_table
    t_0_points, _, _, _, _ =  pc._deformation(
        means3D[deformation_point],
        scales[deformation_point], 
        rotations[deformation_point],
        opacity[deformation_point],
        time[deformation_point]
    )
    means3D_final = torch.zeros_like(means3D)
    means3D_final[deformation_point] =  t_0_points
    means3D_final[~deformation_point] = means3D[~deformation_point]
    breakpoint()
    return means3D_final

def render(
    viewpoint_camera: Camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier = 1.0,
    override_color = None,
    stage="fine",
    # prev_hidden=None,
):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration 
    
    means3D = pc.get_xyz
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=pipe.debug
    )
    time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0], 1)
    force = torch.tensor(viewpoint_camera.force).to(dtype=torch.float32, device=means3D.device).repeat(means3D.shape[0], 1)
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    means2D = screenspace_points
    opacity = pc._opacity
    shs = pc.get_features

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc._scaling
        rotations = pc._rotation

    if "coarse" in stage:
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = means3D, scales, rotations, opacity, shs
        momentum_reg = torch.tensor(0.0)
    else:
        # means3D.shape = (points, 3)
        # scales.shape = (points, 3)
        # rotations.shape = (points, 4)
        # opacity.shape = (points, 1)
        # shs.shape = (points, 16, 3)
        # time.shape = (points, 1)
        # force.shape = (points, 7)
        # points = pc._xyz.shape[0]
        # means3D2 = means3D.repeat([3, 1, 1])
        # scales2 = scales.repeat([3, 1, 1])
        # rotations2 = rotations.repeat([3, 1, 1])
        # opacity2 = opacity.repeat([3, 1, 1])
        # shs2 = shs.repeat([3, 1, 1, 1])
        # time2 = time.repeat([3, 1, 1])
        # force2 = force.repeat([3, 1, 1])
        # means3D_final, scales_final, rotations_final, opacity_final, shs_final \
        #     = pc._deformation.forward(means3D2, scales2, rotations2, opacity2, shs2, time2, force2)
        # print(f"defor table shape: {pc._deformation_table.shape}")
        # breakpoint()
        means3D_final, scales_final, rotations_final, opacity_final, shs_final \
            = pc._deformation.forward(means3D, scales, rotations, opacity, shs, time, force)
        # use magic number 1/39 for time-step difference
        time_prev = torch.ones_like(time) * (viewpoint_camera.time - 1/39)
        time_nxt = torch.ones_like(time) * (viewpoint_camera.time + 1/39)
        # use only means now
        means3D_prev, scales_prev, rotations_prev, opacity_prev, shs_prev \
            = pc._deformation.forward(means3D, scales, rotations, opacity, shs, time_prev, force)
        means3D_nxt, scales_nxt, rotatiosn_nxt, opacity_nxt, shs_nxt \
            = pc._deformation.forward(means3D, scales, rotations, opacity, shs, time_nxt, force)
        momentum_reg = torch.abs(means3D_nxt + means3D_prev - 2 * means3D_final).mean() \
            + torch.abs(opacity_final - opacity_prev).mean() \
            + torch.abs(opacity_final - opacity_nxt).mean()

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

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii, depth = rasterizer.forward(
        means3D = means3D_final,
        means2D = means2D,
        shs = shs_final,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales_final,
        rotations = rotations_final,
        cov3D_precomp = cov3D_precomp
    )

    return rendered_image, screenspace_points, radii > 0, radii, depth, momentum_reg # , hidden

