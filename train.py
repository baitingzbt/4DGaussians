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
from gaussian_renderer.renderer import render
import sys
from scene import Scene, GaussianModel
from scene.cameras import Camera
from utils.general_utils import safe_state
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams, GroupParams
from torch.utils.data import DataLoader
from utils.timer import Timer
import wandb
import math
from utils.scene_utils import render_training_image
from copy import deepcopy
from collections import defaultdict

to8b = lambda x : (255 * np.clip(x.cpu().numpy(), 0, 1)).astype(np.uint8)
FRAMES_EACH = 40
EVAL_EVERY = 10000
SAVE_EVERY = 50000
LOG_EVERY = 500

def scene_reconstruction(
    dataset: GroupParams,
    opt: GroupParams,
    hyper: GroupParams,
    pipe: GroupParams,
    checkpoint: str,
    gaussians: GaussianModel,
    scene: Scene,
    stage: str,
    timer: Timer,
    use_wandb: bool,
    save_video: bool = True,
    save_images: bool = False,
    save_pointclound: bool = True,
    weighted_loss: int = -1
):
    train_iter = opt.coarse_iterations if stage == 'coarse' else opt.iterations
    first_iter = 0
    gaussians.training_setup(opt)
    if checkpoint:
        if stage == "coarse" and stage not in checkpoint:
            print("start from fine stage, skip coarse stage.")
            return
        if stage in checkpoint: 
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, opt)


    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0

    final_iter = train_iter
    if train_iter <= 0 or first_iter > train_iter:
        print("\n---> [no training], possibly wrong specified iterations <---\n")
        return
    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter += 1
    test_cams = scene.getTestCameras()
    train_cams = scene.getTrainCameras()
    n_train_cams = dataset.n_train_cams
    _train_cams = [randint(0, cams-1) + sum(n_train_cams[:i]) for i, cams in enumerate(n_train_cams)]
    train_views = []
    for _cam in _train_cams:
        train_views += [train_cams[_cam * FRAMES_EACH + i] for i in range(FRAMES_EACH)]

    test_views = [test_cams[i] for i in range(len(test_cams))]
    viewpoint_stack = [i for i in tqdm(train_cams, desc='loading data...')]
    viewpoint_force_idx = [cam.force_idx for cam in viewpoint_stack]
    # temp_list = deepcopy(viewpoint_stack)
    n_forces = max(viewpoint_force_idx) + 1
    all_valid_indices = {fi: [] for fi in range(n_forces)}
    for i, fi in enumerate(viewpoint_force_idx):
        all_valid_indices[fi].append(i)
    progress_by_force = np.array([1e6 for fi in range(n_forces)])


    batch_size = opt.batch_size
    timer.start()
    best_psnr = 40.
    for iteration in range(first_iter, final_iter):        

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        shuffle_inter = int(len(train_cams) / batch_size) + 1
        if weighted_loss == -1:
            # shuffle the two lists randomly but keep the same relative order
            # reshuffle after going through entire data once
            if iteration % shuffle_inter == 0 or iteration == first_iter:
                _zipped = list(zip(viewpoint_stack, viewpoint_force_idx))
                random.shuffle(_zipped)
                viewpoint_stack, viewpoint_force_idx = zip(*_zipped)
            # go through dataset sequentially for next <batch_size>
            query_idxs = [((iteration * batch_size) + _i) % len(train_cams) for _i in range(batch_size)]
        ########### hand-tuned weighting deprecated ###########
        # # Don't shuffle. No need to follow each Camera()
        # elif weighted_loss == 0:
        #     # equal prob sample
        #     if stage == 'coarse' or iteration < 80000:
        #         query_force_idx = np.random.randint(0, n_forces)
        #     else:
        #         # easy/hard                          # naive/easy/hard
        #         p = [0.35, 0.65] if n_forces == 2 else [0.25, 0.35, 0.4]
        #         query_force_idx = np.random.choice(n_forces, p=p)
        #     # sample a cam with matching query_force_idx
        #     query_idxs = random.choice(all_valid_indices[query_force_idx])
        elif weighted_loss == 1:
            p = progress_by_force / np.sum(progress_by_force)
            query_force_idx = np.random.choice(n_forces, p=p)
            query_idxs = random.choice(all_valid_indices[query_force_idx])
        
        points = gaussians.get_xyz.shape[0]
        image_tensor = torch.zeros(size=(batch_size, 3, 800, 800), device='cuda')
        gt_image_tensor = torch.zeros(size=(batch_size, 3, 800, 800), device='cuda')
        visibility_tensor = torch.zeros(size=(batch_size, points), device='cuda')
        radii_tensor = torch.zeros(size=(batch_size, points), device='cuda')
        viewspace_point_tensor = [] # torch.zeros(size=(batch_size, points, 3), device='cuda')
        # for j, viewpoint_cam in enumerate(viewpoint_cams):
        #     image, viewspace_point_tensor, visibility_filter, radii, depth \
        #         = render(viewpoint_cam, gaussians, pipe, background, stage=stage)
        #     image_tensor[j] = image
        #     gt_image = viewpoint_cam.original_image.cuda()
        #     gt_image_tensor[j] = gt_image

        total_momentum_reg = 0
        for _i, query_idx in enumerate(query_idxs):
            image, viewspace_point, visibility_filter, radii, depth, momentum_reg \
                = render(viewpoint_stack[query_idx], gaussians, pipe, background, stage=stage)
            image_tensor[_i] = image
            gt_image_tensor[_i] = viewpoint_stack[query_idx].original_image.cuda()
            viewspace_point_tensor.append(viewspace_point)
            radii_tensor[_i] = radii
            visibility_tensor[_i] = visibility_filter
            total_momentum_reg += 0.001 * momentum_reg
        radii = radii_tensor.max(dim=0).values
        visibility_filter = visibility_tensor.any(dim=0)
        # breakpoint()
        # Loss
        loss = l1_loss(image_tensor, gt_image_tensor[:, :3, :, :])
        psnr_ = psnr(image_tensor, gt_image_tensor).mean() # .double()
        # norm
        if stage == "fine":
            l1_reg_weight = hyper.l1_time_planes # * float(iteration > final_iter / 3)
            l2_reg_weight = hyper.l2_time_planes # * float(iteration <= final_iter / 3)
            tv_loss_dict = gaussians.compute_regulation(hyper.time_smoothness_weight, l1_reg_weight, l2_reg_weight, hyper.plane_tv_weight)
            loss += tv_loss_dict['total_reg']
            loss += total_momentum_reg
        else:
            tv_loss_dict = defaultdict(lambda: torch.tensor(0)) # assume to be 0 for coarse

        if opt.lambda_dssim != 0:
            ssim_loss = ssim(image_tensor, gt_image_tensor)
            loss += opt.lambda_dssim * (1.0-ssim_loss)
        
        # if opt.lambda_lpips !=0:
        #     lpipsloss = lpips_loss(image_tensor,gt_image_tensor,lpips_model)
        #     loss += opt.lambda_lpips * lpipsloss

        loss.backward()
        if weighted_loss == 1:
            n_this_force = len(all_valid_indices[query_force_idx])
            progress_by_force[query_force_idx] = \
                ((n_this_force - 1) * progress_by_force[query_force_idx] - math.log10(loss.item())) / n_this_force

        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point)
        for idx in range(batch_size):
            viewspace_point_tensor_grad += viewspace_point_tensor[idx].grad

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log
            total_point = gaussians._xyz.shape[0]
            if iteration % LOG_EVERY == 0:
                # breakpoint()
                infos = {
                    "Loss": round(ema_loss_for_log, 7),
                    "psnr": round(psnr_.item(), 2),
                    "point": total_point,
                    "mom_reg": total_momentum_reg.item()
                }
                for aux_loss_name, aux_loss_val in tv_loss_dict.items():
                    infos[aux_loss_name] = aux_loss_val.item()
                progress_bar.set_postfix({k: str(v) for k, v in infos.items()})
                progress_bar.update(LOG_EVERY)
                # if weighted_loss == 1 and use_wandb:
                #     hist_data = wandb.Histogram(
                #         sample_weight=np.histogram(progress_by_force / np.sum(progress_by_force), bins=[0, 1/3, 2/3, 1])
                #     )
                #     wandb.log({"sampling distribution": hist_data})
                if use_wandb:
                    wandb.log(infos)

            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            timer.pause()
            
            # if dataset.render_process:
            if (iteration % EVAL_EVERY == EVAL_EVERY - 1) \
                or (iteration == final_iter-1):
                # or (psnr_ > best_psnr and stage == 'fine'): # or (iteration == 1):
                name_test = stage+"test"
                name_train = stage+"train"
                # if psnr_ > best_psnr and stage == 'fine':
                #     name_test += '_best'
                #     name_train += '_best'
                #     best_psnr = max(psnr_, best_psnr) + 0.5
                render_training_image(
                    scene, gaussians, test_views, pipe, background, name_test,
                    iteration, timer.get_elapsed_time(), save_video, save_pointclound, save_images, use_wandb)
                render_training_image(
                    scene, gaussians, train_views, pipe, background, name_train,
                    iteration, timer.get_elapsed_time(), save_video, save_pointclound, save_images, use_wandb)
                
            if (iteration % SAVE_EVERY == SAVE_EVERY - 1) or (iteration == final_iter-1):
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                scene.save(iteration, stage)
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" +f"_{stage}_" + str(iteration) + ".pth")

            timer.start()
            # Densification
            if iteration < opt.densify_until_iter :
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter].cuda(), radii[visibility_filter].cuda())
                gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)

                if stage == "coarse":
                    opacity_threshold = opt.opacity_threshold_coarse
                    densify_threshold = opt.densify_grad_threshold_coarse
                else:    
                    opacity_threshold = opt.opacity_threshold_fine_init - iteration * (opt.opacity_threshold_fine_init - opt.opacity_threshold_fine_after) / opt.densify_until_iter
                    densify_threshold = opt.densify_grad_threshold_fine_init - iteration * (opt.densify_grad_threshold_fine_init - opt.densify_grad_threshold_after) / opt.densify_until_iter
                
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 and gaussians.get_xyz.shape[0] < 360000:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold, 5, 5, scene.model_path, iteration, stage)

                if iteration > opt.pruning_from_iter and iteration % opt.pruning_interval == 0 and gaussians.get_xyz.shape[0] > 240000:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.prune(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)
                    
                if iteration % opt.densification_interval == 0 and gaussians.get_xyz.shape[0] < 360000 and opt.add_point:
                    gaussians.grow(5, 5, scene.model_path, iteration,stage)

                if iteration % opt.opacity_reset_interval == 0:
                    print("reset opacity")
                    gaussians.reset_opacity()

        # Optimizer step
        if iteration < opt.iterations:
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none = True)

def training(dataset, hyper, opt, pipe, checkpoint, expname, use_wandb, weighted_loss):
    prepare_output_and_logger(args, expname)
    gaussians = GaussianModel(dataset.sh_degree, hyper)
    dataset.model_path = args.model_path
    timer = Timer()
    scene = Scene(dataset, gaussians)
    scene.model_path = args.model_path
    scene_reconstruction(
        dataset, opt, hyper, pipe, checkpoint, gaussians,
        scene, "coarse", timer,
        use_wandb, save_video=True, save_images=False, save_pointclound=False, weighted_loss=weighted_loss
    )
    scene_reconstruction(
        dataset, opt, hyper, pipe, checkpoint, gaussians,
        scene, "fine", timer,
        use_wandb, save_video=True, save_images=False, save_pointclound=False, weighted_loss=weighted_loss
    )

def prepare_output_and_logger(args, expname: str) -> None:
    if not args.model_path:
        args.model_path = os.path.join("./output/", expname)
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
    torch.cuda.empty_cache()
    torch.set_default_dtype(torch.float32)
    torch.set_default_tensor_type('torch.FloatTensor')
    parser = ArgumentParser(description="Training script parameters")
    setup_seed(6666)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--wandb', action='store_true', default=False)
    parser.add_argument("--quiet", action="store_true", default=False)
    parser.add_argument("--weighted_loss", type=int, default=-1) # -1: none; 0: hand-tuned; 1: dynamic
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--expname", type=str, default = "")
    parser.add_argument("--configs", type=str, default = "")
    parser.add_argument("--data_path_train", type=str, nargs='+', default = [])
    parser.add_argument("--data_path_test", type=str, nargs='+', default = [])
    parser.add_argument("--n_train_cams", type=int, nargs='+', default = [])
    parser.add_argument("--n_test_cams", type=int, nargs='+', default = [])
    
    args = parser.parse_args()
    # args.source_path_train = args.data_path_train
    # args.source_path_test = args.data_path_test
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
        wandb.init(project='4dgs_force', name=args.expname)

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
        args.weighted_loss
    )

    # All done
    print("\nTraining complete.")