import numpy as np
import os
import torch
from scene import Scene, GaussianModel
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams, GroupParams
import wandb
from utils.scene_utils import render_training_image
from tqdm import tqdm


FRAMES_EACH = 40
@torch.no_grad()
def evaluate(dataset, hyper, opt, pipe, checkpoint, expname, use_wandb):
    gaussians = GaussianModel(dataset.sh_degree, hyper)
    dataset.model_path = args.model_path
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    assert checkpoint and "fine" in checkpoint # only use fine-stage ckpt for evaluation
    model_params, first_iter = torch.load(checkpoint)
    gaussians.restore(model_params, opt)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    train_cams = scene.getTrainCameras()
    n_train_cams = dataset.n_train_cams
    assert len(n_train_cams) == 1 # don't load excess data
    assert n_train_cams[0] * FRAMES_EACH == len(train_cams)
    train_views = [train_cams[i] for i in tqdm(range(len(train_cams)), desc='loading train views...')]
    custom_force = [-3, 3, 1.2, 0.78539816, 0.78539816, 0, 100]
    for cam in train_views:
        cam.force= custom_force
    render_training_image(
        scene, gaussians, train_views, pipe, background, f"fine", 0.0,
        first_iter, 0.0, save_video=True, save_pointclound=False, save_images=False, use_wandb=use_wandb
    )


if __name__ == "__main__":
    torch.cuda.empty_cache()
    torch.set_default_dtype(torch.float32)
    torch.set_default_tensor_type('torch.FloatTensor')
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument('--wandb', action='store_true', default=False)
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--expname", type=str, default = "")
    parser.add_argument("--data_path", type=str, nargs='+', default = [])
    parser.add_argument("--configs", type=str, default = "")
    args = parser.parse_args()
    args.source_path = args.data_path
    
    # just read a sample set of camera, then we can manually adjust the time & force
    lp.n_train_cams = [5]
    lp.n_test_cams = [1]
    args.n_train_cams = [5]
    args.n_test_cams = [1]
    args.model_path = os.path.join("./output/", args.expname)
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)

    if args.wandb:
        wandb.init(project='4dgs_force', name=args.expname)

    evaluate(
        lp.extract(args),
        hp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.start_checkpoint,
        args.expname,
        args.wandb,
    )