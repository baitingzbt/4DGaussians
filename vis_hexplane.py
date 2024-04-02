import numpy as np
import torch
from scene import Scene, GaussianModel
import os
import cv2
from tqdm import tqdm
from os import makedirs
from gaussian_renderer.renderer import render
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, ModelHiddenParams, OptimizationParams
from time import time
import open3d as o3d
from utils.render_utils import get_state_at_time
from scene.hexplane import grid_sample_wrapper

ckpt_path = "/sdd/baiting/4DGaussians/output/scene2_directions_full_cont/chkpnt_fine_199999.pth"

@torch.no_grad()
def load_model(dataset: ModelParams, hyperparam: ModelHiddenParams, opt: OptimizationParams):
    gaussians = GaussianModel(dataset.sh_degree, hyperparam)
    gaussians.training_setup(opt)
    scene = Scene(dataset, gaussians)
    (model_params, first_iter) = torch.load(ckpt_path)
    gaussians.restore(model_params, opt)
    return gaussians, scene


parser = ArgumentParser(description="Testing script parameters")
model = ModelParams(parser)
pipeline = PipelineParams(parser)
hyperparam = ModelHiddenParams(parser)
opt = OptimizationParams(parser)
parser.add_argument("--quiet", action="store_true")
parser.add_argument("--configs", type=str)
parser.add_argument("--data_path_train", type=str, nargs='+', default = [])
parser.add_argument("--data_path_test", type=str, nargs='+', default = [])
parser.add_argument("--n_train_cams", type=int, nargs='+', default = [])
parser.add_argument("--n_test_cams", type=int, nargs='+', default = [])
parser.add_argument("--data_drive", type=str, default='/sdd/baiting/4DGaussians/')
args = parser.parse_args()

args.data_path_train = [os.path.join(args.data_drive, p) for p in args.data_path_train]
args.data_path_test = [os.path.join(args.data_drive, p) for p in args.data_path_test]
model.n_train_cams = [int(val) for val in args.n_train_cams]
model.n_test_cams = [int(val) for val in args.n_test_cams]
model.data_path_train = args.data_path_train
model.data_path_test = args.data_path_test
assert len(args.n_train_cams) == len(args.data_path_train)
assert len(args.n_test_cams) == len(args.data_path_test)

if args.configs:
    import mmcv
    from utils.params_utils import merge_hparams
    config = mmcv.Config.fromfile(args.configs)
    args = merge_hparams(args, config)

# Initialize system state (RNG)
safe_state(args.quiet)
gaussians, scene = load_model(model.extract(args), hyperparam.extract(args), opt.extract(args))
points = gaussians.get_xyz

print(123)


# print(gaussians._deformation.deformation_net.grid.aabb.shape)
print(gaussians._deformation.deformation_net.grid.grids)
print(gaussians._deformation.deformation_net.grid.get_density)
# print(gaussians._deformation.deformation_net.grid.grids[0][2])
# breakpoint()


"""
Hexplane initialization:

The input dimensions are [x, y, z, tf] with 6 combinations being
    [(x, y) (x, z) (x, tf) (y, z) (y, tf) (z, tf)]
    [(0, 1) (0, 2) (0, 3)  (1, 2) (1, 3) (2, 3)]
    [   0      1      2       3      4      5  ]

So the ones related with <tf> are number [2, 4, 5] and corresponding tuples
"""


tf_grids = [2, 4, 5]
tf_combs = [(0, 3), (1, 3), (2, 3)]

xyz_grids = [0, 1, 3]
xyz_combs = [(0, 1), (0, 2), (1, 2)]




if False:

    # first 3-dim taken from a random training point, 4-th dim is reserved for force-time
    sample_pt = torch.tensor([[0.4493, -0.5141, -0.2839, 0.0]], device='cuda', dtype=torch.float32)
    # a force-tensor taken from training data
    force_tensor = torch.tensor([[1.3090, 0.2618, 0.0000, 400]], device='cuda', dtype=torch.float32)
    time_tensor = torch.tensor([[1/39]], device='cuda', dtype=torch.float32)
    force_emb = torch.exp(gaussians._deformation.deformation_net.force_embedder(force_tensor / 10))
    force_time_emb = gaussians._deformation.deformation_net.force_time_embed(torch.cat((time_tensor, force_emb), dim=1))
    print("force_time_emb: ", force_time_emb)
    sample_pt[:, 3] = force_time_emb

    interp_space = 1.
    for ci, coo_comb in zip(tf_grids, tf_combs):
        grid = gaussians._deformation.deformation_net.grid.grids[0][ci]
        feature_dim = grid.shape[1]
        intep_out = grid_sample_wrapper(grid, sample_pt[..., coo_comb]).view(-1, feature_dim)
        interp_space *= intep_out
    print(interp_space)


    # below is example showing xyz-plane has a much bigger impact on inter_space
    xyz_grids = [0, 1, 3]
    xyz_combs = [(0, 1), (0, 2), (1, 2)]
    interp_space = 1.
    for ci, coo_comb in zip(xyz_grids, xyz_combs):
        grid = gaussians._deformation.deformation_net.grid.grids[0][ci]
        feature_dim = grid.shape[1]
        intep_out = grid_sample_wrapper(grid, sample_pt[..., coo_comb]).view(-1, feature_dim)
        interp_space *= intep_out
    print(interp_space)



if True:
    # we try out a few different forces and different times, with fixec xyz
    # we get the corresponding embedding, and see how influential theyr are
    # in the hexplane
    sample_pt = torch.tensor([[0, 0, 0, 0]], device='cuda', dtype=torch.float32)
    sample_pt[:, :3] = points[0]
    forces_collection = [
        [-1.5707963267948966, 0.0, 0.0, 250.0],
        [1.5707963267948966, 0.0, 0.0, 250.0],
        [0.2617993877991494, 1.3089969389957472, 0.0, 250.0],
        [-0.7853981633974483, 0.7853981633974483, 0.0, 250.0],
        # # changing intensity (notice this is a novel direction setting)
        # [0.2617993877991494, 1.3089969389957472, 0.0, 350.0],
        # [1.5707963267948966, 0.0, 0.0, 350.0],
    ]


    for force in forces_collection:
        force_tensor = torch.tensor([force], dtype=torch.float32, device='cuda')
        for t in range(0, 5):
            time_tensor = torch.tensor([[1/39]], dtype=torch.float32, device='cuda') * t
            force_emb = torch.exp(gaussians._deformation.deformation_net.force_embedder(force_tensor / 10))
            force_time_emb = gaussians._deformation.deformation_net.force_time_embed(torch.cat((time_tensor, force_emb), dim=1))
            sample_pt[:, 3] = force_time_emb

            # time-force-grid value
            interp_space_tf = 1.
            for ci, coo_comb in zip(tf_grids, tf_combs):
                grid = gaussians._deformation.deformation_net.grid.grids[0][ci]
                feature_dim = grid.shape[1]
                intep_out = grid_sample_wrapper(grid, sample_pt[..., coo_comb]).view(-1, feature_dim)
                interp_space_tf *= intep_out

            # xyz-grid value
            interp_space_xyz = 1.
            for ci, coo_comb in zip(xyz_grids, xyz_combs):
                grid = gaussians._deformation.deformation_net.grid.grids[0][ci]
                feature_dim = grid.shape[1]
                intep_out = grid_sample_wrapper(grid, sample_pt[..., coo_comb]).view(-1, feature_dim)
                interp_space_xyz *= intep_out

            print(f"\nforce = {force}, time = {t}/39")
            print(f"\ttime-force final: {torch.mean(interp_space_tf)}")
            print(f"\txyz final: {torch.mean(interp_space_xyz)}")



if True:
    # change point, keep time-force the same
    for pt in points[:5]:
        sample_pt = torch.tensor([[0, 0, 0, 0]], dtype=torch.float32, device='cuda')
        sample_pt[:, :3] = pt
        force_tensor = torch.tensor([[-1.5707963267948966, 0.0, 0.0, 250.0]], dtype=torch.float32, device='cuda')
        time_tensor = torch.tensor([[1/39]], dtype=torch.float32, device='cuda')
        force_emb = torch.exp(gaussians._deformation.deformation_net.force_embedder(force_tensor / 10))
        force_time_emb = gaussians._deformation.deformation_net.force_time_embed(torch.cat((time_tensor, force_emb), dim=1))
        sample_pt[:, 3] = force_time_emb

        # time-force-grid value
        interp_space_tf = 1.
        for ci, coo_comb in zip(tf_grids, tf_combs):
            grid = gaussians._deformation.deformation_net.grid.grids[0][ci]
            feature_dim = grid.shape[1]
            intep_out = grid_sample_wrapper(grid, sample_pt[..., coo_comb]).view(-1, feature_dim)
            interp_space_tf *= intep_out

        # xyz-grid value
        interp_space_xyz = 1.
        for ci, coo_comb in zip(xyz_grids, xyz_combs):
            grid = gaussians._deformation.deformation_net.grid.grids[0][ci]
            feature_dim = grid.shape[1]
            intep_out = grid_sample_wrapper(grid, sample_pt[..., coo_comb]).view(-1, feature_dim)
            interp_space_xyz *= intep_out

        print(f"\nforce = {time_tensor}, time = 1/39, pt={pt}")
        print(f"\ttime-force final: {torch.mean(interp_space_tf)}")
        print(f"\txyz final: {torch.mean(interp_space_xyz)}")

