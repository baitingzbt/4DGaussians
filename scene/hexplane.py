import itertools
import logging as log
from typing import Optional, Union, List, Dict, Sequence, Iterable, Collection, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

def get_normalized_directions(directions):
    """SH encoding must be in the range [0, 1]

    Args:
        directions: batch of directions
    """
    return (directions + 1.0) / 2.0

def normalize_aabb(pts, aabb):
    return (pts - aabb[0]) * (2.0 / (aabb[1] - aabb[0])) - 1.0

def grid_sample_wrapper(grid: torch.Tensor, coords: torch.Tensor, align_corners: bool = True) -> torch.Tensor:
    # breakpoint()
    grid_dim = coords.shape[-1]
    if grid.dim() == grid_dim + 1: # NOTE: not entering this branch
        # no batch dimension present, need to add it
        grid = grid.unsqueeze(0)
    if coords.dim() == 2: # NOTE: entering this branch
        coords = coords.unsqueeze(0)
    if grid_dim == 2 or grid_dim == 3: # NOTE: entering this branch
        grid_sampler = F.grid_sample
    else:
        raise NotImplementedError(f"Grid-sample was called with {grid_dim}D data but is only "
                                  f"implemented for 2 and 3D data.")

    coords = coords.view([coords.shape[0]] + [1] * (grid_dim - 1) + list(coords.shape[1:]))
    B, feature_dim = grid.shape[:2]
    n = coords.shape[-2]
    interp = grid_sampler(
        grid,  # [B, feature_dim, reso, ...]
        coords,  # [B, 1, ..., n, grid_dim]
        align_corners=align_corners,
        mode='bilinear',
        padding_mode='border'
    )
    interp = interp.view(B, feature_dim, n).transpose(-1, -2)  # [B, n, feature_dim]
    interp = interp.squeeze()  # [B?, n, feature_dim?]
    return interp

def init_grid_param(
    grid_nd: int,
    in_dim: int,
    out_dim: int,
    reso: Sequence[int],
    a: float = 0.5,
    b: float = 1.5
):
    # print(f"grid_nd = {grid_nd}\nin_dim = {in_dim}\nout_dim = {out_dim}\nlen(reso) = {len(reso)}")
    # breakpoint()
    assert in_dim == len(reso), "Resolution must have same number of elements as input-dimension"
    has_time_planes = bool(in_dim >= 4)
    has_force_planes = bool(in_dim >= 5)
    assert grid_nd <= in_dim
    # if using force and not blend with time, <coo_combs> looks like
    # 0: x, y
    # 1: x, z
    # 2: x, t
    # 3: x, f
    # 4: y, z
    # 5: y, t
    # 6: y, f
    # 7: z, t
    # 8: z, f
    # 9: t, f -> dropped
    # related to t: (2, 5, 7)     |    related to f: (3, 6, 8)
    coo_combs = list(itertools.combinations(range(in_dim), grid_nd))
    # drop plane for (f, t) (3, 4) if it exists
    if len(coo_combs) == 10:
        coo_combs = coo_combs[:-1]
    grid_coefs = nn.ParameterList()
    for ci, coo_comb in enumerate(coo_combs):
        new_grid_coef = nn.Parameter(
            torch.empty(
                [1, out_dim] + [reso[cc] for cc in coo_comb[::-1]]
            )
        )
        # x, y, z, time, force;
        # below means if time or force exist in current coo_comb, initialize those planes to 1

        # this is saying, if blend force+time, then don't init anything as ones_
        # but if not blending force and time, we init the ones related to time (xt, yt, zt) as ones_
        # and keep the (xf, yf, zf) still as uniform_
        # if max(coo_comb) == 3 and has_force_planes and has_time_planes:
        #     nn.init.ones_(new_grid_coef)
        nn.init.uniform_(new_grid_coef, a=a, b=b)
        grid_coefs.append(new_grid_coef)

    return grid_coefs


def interpolate_ms_features(
    pts: torch.Tensor,
    ms_grids: Collection[Iterable[nn.Module]],
    grid_dimensions: int,
    concat_features: bool
) -> torch.Tensor:
    coo_combs = list(itertools.combinations(range(pts.shape[-1]), grid_dimensions))
    # drop plane for (f, t) (3, 4) if it exists
    if len(coo_combs) == 10:
        coo_combs = coo_combs[:-1]
    multi_scale_interp = [] if concat_features else 0.
    grid: nn.ParameterList
    for scale_id, grid in enumerate(ms_grids):
        interp_space = 1.
        for ci, coo_comb in enumerate(coo_combs):
            # interpolate in plane
            feature_dim = grid[ci].shape[1]  # shape of grid[ci]: 1, out_dim, *reso
            interp_out_plane = (grid_sample_wrapper(grid[ci], pts[..., coo_comb]).view(-1, feature_dim))
            # compute product over planes
            # NOTE: if the grid[ci] of <coo_combs> actually matters, interp_out_plane should change more?
            # this means
            interp_space = interp_space * interp_out_plane
            # print(interp_space.mean().item())
        # combine over scales
        if concat_features:
            multi_scale_interp.append(interp_space)
        else:
            multi_scale_interp = multi_scale_interp + interp_space

    if concat_features:
        multi_scale_interp = torch.cat(multi_scale_interp, dim=-1)
    return multi_scale_interp


class HexPlaneField(nn.Module):

    def __init__(
        self,
        bounds,
        planeconfig,
        multires,
        hidden_width
    ) -> None:
        super().__init__()
        # breakpoint()
        aabb = torch.tensor([
            [bounds, bounds, bounds],
            [-bounds,-bounds, -bounds]
        ])
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        self.grid_config = [planeconfig]
        self.multiscale_res_multipliers = multires
        self.concat_features = True
        # 1. Init planes
        self.grids = nn.ModuleList()
        self.feat_dim = 0
        for res in tqdm(self.multiscale_res_multipliers, desc = 'Initializing HexPlaneField Layers'):
            # initialize coordinate grid
            config = self.grid_config[0].copy()
            # Resolution fix: multi-res only on spatial planes
            config["resolution"] = [
                r * res for r in config["resolution"][:3]
            ] + config["resolution"][3:]
            gp = init_grid_param(
                grid_nd=config["grid_dimensions"],
                in_dim=config["input_coordinate_dim"],
                out_dim=config["output_coordinate_dim"],
                reso=config["resolution"],
            )
            # shape[1] is out-dim - Concatenate over feature len for each scale
            if self.concat_features:
                self.feat_dim += gp[-1].shape[1]
            else:
                self.feat_dim = gp[-1].shape[1]
            self.grids.append(gp)

        self.hidden_width = hidden_width
        self.recur_feat_dim = int(self.feat_dim + self.hidden_width)
        self.feature_aggr = nn.Sequential(
            nn.Linear(self.recur_feat_dim, self.recur_feat_dim),
            nn.ReLU(),
            nn.Linear(self.recur_feat_dim, self.feat_dim)
        )

    @property
    def get_aabb(self):
        return self.aabb[0], self.aabb[1]
    
    def set_aabb(self, xyz_max, xyz_min):
        aabb = torch.tensor([xyz_max, xyz_min], dtype=torch.float32)
        self.aabb = nn.Parameter(aabb,requires_grad=False)

    def get_density(
        self,
        pts: torch.Tensor,
        time: Optional[torch.Tensor] = None,
        force: Optional[torch.Tensor] = None
    ):
        """Computes and returns the densities."""
        pts = normalize_aabb(pts, self.aabb)
        if time is not None:
            pts = torch.cat((pts, time), dim=-1)  # [n_rays, n_samples, 4 OR 10]
        if force is not None:
            pts = torch.cat((pts, force), dim=-1)
        pts = pts.reshape(-1, pts.shape[-1])
        
        features = interpolate_ms_features(
            pts,
            ms_grids=self.grids,  # noqa
            grid_dimensions=self.grid_config[0]["grid_dimensions"],
            concat_features=self.concat_features
        )
        
        if len(features) < 1:
            features = torch.zeros((0, 1)).to(features.device)
        return features

    # NOTE: using MLP can help to increase the influence of the time/force-related inputs
    # as in Hexplane, we have to mostly put it as (x, y, z, t, (f)) but now
    def get_features_mlp(self):
        pass

    def forward(
        self,
        pts: torch.Tensor,
        time: Optional[torch.Tensor] = None,
        force: Optional[torch.Tensor] = None
    ):
        return self.get_density(pts, time, force)
