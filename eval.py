import numpy as np
import os
import torch
from scene import Scene, GaussianModel
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams, GroupParams
import wandb
from utils.scene_utils import render_training_image
from tqdm import tqdm
from train import prepare_output_and_logger


FRAMES_EACH = 40
@torch.no_grad()
def evaluate(dataset, hyper, opt, pipe, checkpoint, expname):
    prepare_output_and_logger(args, expname)
    gaussians = GaussianModel(dataset.sh_degree, hyper)
    dataset.model_path = args.model_path
    scene = Scene(dataset, gaussians)
    scene.model_path = args.model_path
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
    
    
    # custom_force = [-3, 3, 1.2, 0.78539816, 0.78539816, 0, 100]

    # i = 0
    # for x in range(-90, 91):
    #     for y in range(-90, 91):
    #         if abs(x) + abs(y) != 90:
    #             continue
    #         if x == -56 and y == 34:
    #             i += 1
    #             custom_force = np.array([np.radians(x), np.radians(y)])
    #             for cam in train_cams:
    #                 cam.force= custom_force
    #             scene.model_path += f"/{i}"
    #             render_training_image(
    #                 scene, gaussians, train_cams, pipe, background, f"fine", first_iter, 0.0,
    #                 save_video=True, save_pointclound=False, save_images=False, use_wandb=False
    #             )
    #             scene.model_path = scene.model_path[:-2]
    all_cams = scene.getTestCameras()
    set1 = [cam for cam in all_cams if int(np.degrees(cam.full_force[0])) % 45 == 0] # in [-90, -45, 0, 45, 90]
    set2 = [cam for cam in all_cams if abs(int(np.degrees(cam.full_force[0])) % 45) <= 10 and abs(int(np.degrees(cam.full_force[0])) % 45) != 0]
    set3 = [cam for cam in all_cams if abs(int(np.degrees(cam.full_force[0])) % 45) > 10]

    # set1 = [cam for cam in all_cams if int(np.degrees(cam.force * 720))] # in [-90, -45, 0, 45, 90]
    # set2 = [cam for cam in all_cams if int(np.degrees(cam.force * 720))]
    # set3 = [cam for cam in all_cams if int(np.degrees(cam.force * 720))]
    # breakpoint()
    render_training_image(
        scene, gaussians, set1, pipe, background, "fine_train", first_iter, 0.0,
        save_video=True, save_pointclound=False, save_images=False, use_wandb=False
    )
    render_training_image(
        scene, gaussians, set2, pipe, background, "fine_test_sim", first_iter, 0.0,
        save_video=True, save_pointclound=False, save_images=False, use_wandb=False
    )
    render_training_image(
        scene, gaussians, set3, pipe, background, "fine_test_far", first_iter, 0.0,
        save_video=True, save_pointclound=False, save_images=False, use_wandb=False
    )
    # render_training_image(
    #     scene, gaussians, all_cams, pipe, background, "fine_test_all", first_iter, 0.0,
    #     save_video=True, save_pointclound=False, save_images=False, use_wandb=False
    # )


if __name__ == "__main__":
    torch.cuda.empty_cache()
    torch.set_default_dtype(torch.float32)
    torch.set_default_tensor_type('torch.FloatTensor')
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--expname", type=str, default = "")
    parser.add_argument("--data_drive", type=str, default='.')
    parser.add_argument("--data_path_train", type=str, nargs='+', default = [])
    parser.add_argument("--data_path_test", type=str, nargs='+', default = [])
    parser.add_argument("--n_train_cams", type=int, nargs='+', default = [])
    parser.add_argument("--n_test_cams", type=int, nargs='+', default = [])
    parser.add_argument("--configs", type=str, default = "")
    args = parser.parse_args()
    args.data_path_train = [os.path.join(args.data_drive, p) for p in args.data_path_train]
    args.data_path_test = [os.path.join(args.data_drive, p) for p in args.data_path_test]
    lp.n_train_cams = [int(val) for val in args.n_train_cams]
    lp.n_test_cams = [int(val) for val in args.n_test_cams]
    lp.data_path_train = args.data_path_train
    lp.data_path_test = args.data_path_test
    assert len(args.n_train_cams) == len(args.data_path_train)
    assert len(args.n_test_cams) == len(args.data_path_test)


    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    evaluate(
        lp.extract(args),
        hp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.start_checkpoint,
        args.expname
    )