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
import numpy as np
import random
import os, sys
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss, lpips_loss
from gaussian_renderer.renderer import render, ANCHORS, render_for_state
import sys
from scene import Scene, GaussianModel
from scene.cameras import Camera
from utils.general_utils import safe_state
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from torch.utils.data import DataLoader
from utils.timer import Timer
import wandb
import math
from utils.scene_utils import render_training_image
from collections import defaultdict
from typing import List
from scene.dataset_readers import START_FRAME, MAX_FRAME

EVAL_ONLY = False
FRAMES_EACH = MAX_FRAME - START_FRAME
EVAL_EVERY = 2000 # 10000
SAVE_EVERY = 20000
LOG_EVERY = 100 # 500
MAX_PHASE = 8
RECUR = True
# CUDA_VISIBLE_DEVICES=1 nohup python train.py --data_path_train data_new/scene3_60 --data_path_test data_new/scene3_60 --n_train_cams 540 --n_test_cams 25 --expname prevframes_pocfre_newforce_onestep_sep --configs arguments/dnerf/force_blend.py --onestep --wandb > prevframes_pocfre_newforce_onestep_sep.out
# this function gets means3D to cameras, which later can be used for dynamic training?
@torch.no_grad()
def get_state(cameras: List[Camera], gaussians: GaussianModel) -> List[Camera]:
    for cam in cameras:
        # print(f"{cam.time = } | base = {START_FRAME / FRAMES_EACH}")
        # use coarse means3D (works when switching cameras)
        if cam.time == START_FRAME / FRAMES_EACH:
            cam.prev_state = None
        # use means3D from last calculation
        else:
            cam.prev_state = state
        state = render_for_state(cam, gaussians)
    return cameras

def filter_cams(cameras: List[Camera], phase: int) -> List[Camera]:
    return [cam for cam in cameras if cam.frame_step % MAX_PHASE <= phase]

def filter_cams2(cameras: List[Camera]) -> List[Camera]:
    return [cam for cam in cameras if cam.frame_step % MAX_PHASE in [2, 5]]

def scene_reconstruction(
    dataset: ModelParams,
    opt: OptimizationParams,
    hyper: ModelHiddenParams,
    pipe: PipelineParams,
    checkpoint: str,
    gaussians: GaussianModel,
    scene: Scene,
    stage: str,
    timer: Timer,
    use_wandb: bool,
    save_video: bool = True,
    save_pointclound: bool = True,
    phase: int = None,
    onestep: bool = False,
    pre_frames: int = 0
):
    pipe.debug = True
    if stage == "coarse":
        train_iter = opt.coarse_iterations
    elif stage == "anchor":
        train_iter = opt.anchor_iterations
    elif stage == "fine":
        train_iter = opt.iterations
    
    first_iter = 0
    gaussians.training_setup(opt)
    if checkpoint:
        if stage == "coarse" and stage not in checkpoint:
            print("start from fine stage, skip coarse stage.")
            return
        if stage in checkpoint: 
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, opt)


    background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda") * dataset.white_background
    ema_loss_for_log = 0.0

    final_iter = train_iter
    if train_iter <= 0 or first_iter > train_iter:
        print("\n---> [no training], possibly wrong specified iterations <---\n")
        return
    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter += 1
    test_cams = scene.getTestCameras()
    train_cams = scene.getTrainCameras()

    if EVAL_ONLY:
        avg_l1_test, avg_psnr_test = render_training_image(
            scene, gaussians, test_cams, pipe, background, stage+"test",
            first_iter, 0, True, False, False)
        avg_l1_train, avg_psnr_train = render_training_image(
            scene, gaussians, train_cams[:10*FRAMES_EACH], pipe, background, stage+"train",
            first_iter, 0, True, False, False)
        exit()

    # n_train_cams = dataset.n_train_cams

    # when multiple trajectories, select one train view on each trajectory at random
    # _train_cams_selected = [randint(0, cams-1) + sum(n_train_cams[:i]) for i, cams in enumerate(n_train_cams)]

    # help to get the most updated train views for dynamic model
    # def update_traincams(cams_selected: List[int]) -> List[Camera]:
    #     train_views = []
    #     for _cam in cams_selected:
    #         train_views += [train_cams[_cam * FRAMES_EACH + i] for i in range(FRAMES_EACH)]
    #     return train_views

    batch_size = opt.batch_size
    timer.start()
    # train_views = update_traincams(_train_cams_selected)

    if "coarse" in stage:
        train_cams = [cam for cam in train_cams if cam.frame_step == START_FRAME]
        print(f"Coarse using time={START_FRAME / FRAMES_EACH} cameras, total of {len(train_cams)} cameras")
    elif onestep:
        train_cams = filter_cams2(train_cams)

    n_train_views = len(train_cams)
    for iteration in range(first_iter, final_iter):        

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # if (iteration % 100 == 0 and RECUR_STATE \
        #     and ("coarse" not in stage) \
        #     and (iteration > opt.densify_until_iter or iteration < opt.densify_from_iter)):
        #     train_cams = get_state(train_cams, gaussians)
        
        points = gaussians.get_xyz.shape[0]
        image_tensor = torch.zeros(size=(batch_size, 3, 480, 480), device='cuda')
        gt_image_tensor = torch.zeros(size=(batch_size, 3, 480, 480), device='cuda')
        visibility_tensor = torch.zeros(size=(batch_size, points), device='cuda')
        radii_tensor = torch.zeros(size=(batch_size, points), device='cuda')
        viewspace_point_tensor = []
        total_momentum_reg = 0
        total_knn_reg = 0
        total_shs_reg = 0
        total_scales_reg = 0
        query_idxs = np.random.choice(range(1, n_train_views-1), batch_size, replace=False)
        for _i, query_idx in enumerate(query_idxs):
            _query_idx = query_idx
            if train_cams[query_idx].frame_step == START_FRAME and ("coarse" not in stage) and not onestep:
                _query_idx = query_idx + 1
            elif train_cams[query_idx].frame_step == MAX_FRAME - 1 and ("coarse" not in stage) and not onestep:
                _query_idx = query_idx - 1
            # if coarse, just use nearby camera as placeholders, their stored data won't be actually used
            train_views = [train_cams[_query_idx-1], train_cams[_query_idx], train_cams[_query_idx+1]]
            image, viewspace_point, visibility_filter, radii, depth, reg_dict \
                = render(train_views, gaussians, pipe, background, stage=stage)
            image_tensor[_i] = image
            gt_image_tensor[_i] = train_views[1].original_image.cuda()
            viewspace_point_tensor.append(viewspace_point)
            radii_tensor[_i] = radii
            visibility_tensor[_i] = visibility_filter
            total_momentum_reg += 0.0001 * reg_dict['momentum']
            total_knn_reg += reg_dict['knn']
            total_shs_reg += 0.001 * reg_dict['shs']
            total_scales_reg += 0.001 * reg_dict['scales']
            

        radii = radii_tensor.max(dim=0).values
        visibility_filter = visibility_tensor.any(dim=0)

        # Loss
        loss = l1_loss(image_tensor, gt_image_tensor[:, :3, :, :])
        psnr_ = psnr(image_tensor, gt_image_tensor).mean()
        # norm
        if stage in ["fine", "anchor"]:
            l1_reg_weight = hyper.l1_time_planes
            l2_reg_weight = hyper.l2_time_planes
            time_weight = hyper.time_smoothness_weight
            plane_weight = hyper.plane_tv_weight
            force_weight = hyper.force_weight
            tv_loss_dict = gaussians.compute_regulation(
                time_weight, l1_reg_weight, l2_reg_weight, plane_weight, force_weight
            )
            loss += tv_loss_dict['total_reg']
            loss += total_momentum_reg * float(total_momentum_reg > 2e-5)
            loss += total_knn_reg
            loss += total_shs_reg
            loss += total_scales_reg
        else:
            tv_loss_dict = defaultdict(lambda: torch.tensor(0)) # assume to be 0 for coarse


        loss.backward()
        # print(f"loss final = {loss}, momentum: {total_momentum_reg}")
        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point)
        for idx in range(batch_size):
            viewspace_point_tensor_grad += viewspace_point_tensor[idx].grad

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % LOG_EVERY == 0:
                infos = {
                    "Loss": round(ema_loss_for_log, 7),
                    "psnr": round(psnr_.item(), 2),
                    "point": points,
                    "mom_reg": total_momentum_reg.item(),
                    "knn_reg": total_knn_reg.item(),
                    "shs_reg": total_shs_reg.item(),
                    "scales_reg": total_scales_reg.item()
                }
                for aux_loss_name, aux_loss_val in tv_loss_dict.items():
                    infos[aux_loss_name] = aux_loss_val.item()
                # progress_bar.set_postfix({k: str(v) for k, v in infos.items()})
                progress_bar.set_postfix({
                    'pts': str(points),
                    'loss': str(round(loss.item(), 5)),
                    'mom': str(round(total_momentum_reg.item(), 7))
                })
                progress_bar.update(LOG_EVERY)
                if use_wandb:
                    wandb.log(infos)

            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            timer.pause()

            if (iteration % SAVE_EVERY == SAVE_EVERY - 1) or (iteration == final_iter-1):
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                scene.save(iteration, stage)
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" +f"_{stage}_" + str(iteration) + ".pth")
            
            if (iteration % EVAL_EVERY == EVAL_EVERY - 1) or (iteration == final_iter-1):
                # if RECUR_STATE and ("coarse" not in stage):
                #     train_cams = get_state(train_cams, gaussians)
                #     test_cams = get_state(test_cams, gaussians)
                # if RECUR_HIDDEN and ("coarse" not in stage):
                #     train_cams = get_hidden(train_cams, gaussians)
                #     test_cams = get_hidden(test_cams, gaussians)
                avg_l1_test, avg_psnr_test = render_training_image(
                    scene, gaussians, test_cams, pipe, background, stage+"test",
                    iteration, timer.get_elapsed_time(), save_video, save_pointclound, use_wandb)
                avg_l1_train, avg_psnr_train = render_training_image(
                    scene, gaussians, train_cams[:FRAMES_EACH * 3], pipe, background, stage+"train",
                    iteration, timer.get_elapsed_time(), save_video, save_pointclound, use_wandb)
                if use_wandb:
                    infos = {
                        "Render-Train-Loss": round(avg_l1_train, 7),
                        "Render-Train-PSNR": round(avg_psnr_train, 2),
                        "Render-Test-Loss": round(avg_l1_test, 7),
                        "Render-Test-PSNR": round(avg_psnr_test, 2),
                    }
                    wandb.log(infos)

            timer.start()
            # Densification
            if iteration < opt.densify_until_iter: # and (stage == "coarse" or not RECUR):
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter].cuda(), radii[visibility_filter].cuda())
                gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)

                if stage == "coarse":
                    opacity_threshold = opt.opacity_threshold_coarse
                    densify_threshold = opt.densify_grad_threshold_coarse
                else:    
                    opacity_threshold = opt.opacity_threshold_fine_init - iteration * (opt.opacity_threshold_fine_init - opt.opacity_threshold_fine_after) / opt.densify_until_iter
                    densify_threshold = opt.densify_grad_threshold_fine_init - iteration * (opt.densify_grad_threshold_fine_init - opt.densify_grad_threshold_after) / opt.densify_until_iter
                
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 and gaussians.get_xyz.shape[0] < 14000:
                    # print("densify")
                    size_threshold = 40 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold, 5, 5, scene.model_path, iteration, stage)

                # don't prune if it is RECUR training
                if iteration > opt.pruning_from_iter and iteration % opt.pruning_interval == 0 and gaussians.get_xyz.shape[0] > 3000 and not RECUR:
                    print("prune")
                    size_threshold = 40 if iteration > opt.opacity_reset_interval else None
                    gaussians.prune(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)
                    
                if iteration % opt.densification_interval == 0 and gaussians.get_xyz.shape[0] < 14000 and opt.add_point:
                    # print("grow")
                    gaussians.grow(5, 5, scene.model_path, iteration, stage)

                if iteration % opt.densification_interval == 0 and ('coarse' not in stage) and RECUR:
                    train_cams = get_state(train_cams, gaussians)

                if iteration % opt.opacity_reset_interval == 0:
                    print("reset opacity")
                    gaussians.reset_opacity()

        # Optimizer step
        if iteration < opt.iterations:
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none = True)

def training(dataset, hyper, opt, pipe, checkpoint, expname, use_wandb, anchors, phase, onestep, prev_frames):
    prepare_output_and_logger(args, expname)
    dataset.model_path = args.model_path
    dataset.prev_frames = prev_frames
    hyper.prev_frames = prev_frames
    gaussians = GaussianModel(dataset.sh_degree, hyper)
    gaussians._deformation.recur = RECUR
    timer = Timer()
    scene = Scene(dataset, gaussians)
    scene.model_path = args.model_path
    scene_reconstruction(
        dataset, opt, hyper, pipe, checkpoint, gaussians,
        scene, "coarse", timer,
        use_wandb, save_video=True, save_pointclound=False,
        phase=None, onestep=False
    )
    # if anchors:
    #     scene_reconstruction(
    #         dataset, opt, hyper, pipe, checkpoint, gaussians,
    #         scene, "anchor", timer,
    #         use_wandb, save_video=True, save_pointclound=False, phase=None
    #     )
    if phase:
        for i in range(START_FRAME, MAX_PHASE):
            scene_reconstruction(
                dataset, opt, hyper, pipe, checkpoint, gaussians,
                scene, "fine", timer,
                use_wandb, save_video=True, save_pointclound=False,
                phase=i, onestep=False
            )
    scene_reconstruction(
        dataset, opt, hyper, pipe, checkpoint, gaussians,
        scene, "fine", timer,
        use_wandb, save_video=True, save_pointclound=False,
        phase=None, onestep=onestep
    )

def prepare_output_and_logger(args, expname: str) -> None:
    if not args.model_path:
        args.model_path = os.path.join(args.data_drive, "output", expname)
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    # torch.cuda.empty_cache()
    torch.set_default_dtype(torch.float32)
    torch.set_default_tensor_type('torch.FloatTensor')
    parser = ArgumentParser(description="Training script parameters")
    setup_seed(42)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--wandb', action='store_true', default=False)
    parser.add_argument("--quiet", action="store_true", default=False)
    parser.add_argument("--anchors", action="store_true", default=False)
    parser.add_argument("--phases", action="store_true", default=False)
    parser.add_argument("--onestep", action="store_true", default=False)
    parser.add_argument("--prev_frames", type=int, default=0)
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--expname", type=str, default = "")
    parser.add_argument("--configs", type=str, default = "")
    parser.add_argument("--data_drive", type=str, default='.')
    parser.add_argument("--data_path_train", type=str, nargs='+', default = [])
    parser.add_argument("--data_path_test", type=str, nargs='+', default = [])
    parser.add_argument("--n_train_cams", type=int, nargs='+', default = [])
    parser.add_argument("--n_test_cams", type=int, nargs='+', default = [])
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

    if args.wandb:
        wandb.init(project='4dgs_force', name=args.expname, dir=args.data_drive, settings=wandb.Settings(start_method="fork"))

    # Initialize system state (RNG)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        hp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.start_checkpoint,
        args.expname,
        args.wandb,
        args.anchors,
        args.phases,
        args.onestep,
        args.prev_frames
    )

    # All done
    print("\nTraining complete.")