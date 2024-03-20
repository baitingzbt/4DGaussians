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


ckpt_path = "/sdd/baiting/4DGaussians/output/scene2_directions_full_cont/chkpnt_fine_199999.pth"

@torch.no_grad()
def load_model(dataset: ModelParams, hyperparam: ModelHiddenParams, opt: OptimizationParams):
    gaussians = GaussianModel(dataset.sh_degree, hyperparam)
    scene = Scene(dataset, gaussians)
    (model_params, first_iter) = torch.load(ckpt_path)
    gaussians.restore(model_params, opt)
    return gaussians, scene


parser = ArgumentParser(description="Testing script parameters")
model = ModelParams(parser, sentinel=True)
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


print(123)


# gaussians._deformation.deformation_net.grid