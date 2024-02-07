_base_ = './dnerf_default.py'

USE_FORCE = True
USE_TIME = True
BLEND_TIME_FORCE = True
RESOLUTION = [64, 64, 64]
INPUT_DIM = 3

if USE_TIME:
    RESOLUTION += [128] # 64, 128
    INPUT_DIM += 1

if USE_FORCE and not BLEND_TIME_FORCE:
    RESOLUTION += [64] # [32, 32, 32, 64]
    INPUT_DIM += 1 # 4

OptimizationParams = dict(
    coarse_iterations = 12000, # default: 3000
    iterations = 500000
)

ModelParams = dict(
    extension='',
)

ModelHiddenParams = dict(
    kplanes_config = {
        'grid_dimensions': 2, # default: 2
        'input_coordinate_dim': INPUT_DIM,
        'output_coordinate_dim': 64, # 64, 128
        'resolution': RESOLUTION
    },
    # bounds=2.0,
    defor_depth = 4,
    use_force = USE_FORCE,
    use_time = USE_TIME,
    blend_time_force = BLEND_TIME_FORCE,
    plane_tv_weight = 0.0001,
    time_smoothness_weight = 0.01, # default: 0.01
    l1_time_planes = 0.0001, # default: 0.0001
    l2_time_planes = 0., # 0005,
    multires = [1, 2]
)
# CUDA_VISIBLE_DEVICES=2 nohup python train.py --data_path data_new/scene2_force1 data_new/scene2_force2 --expname "force_scene2_blend2" --configs arguments/dnerf/force.py --multi_cam --wandb > force_scene2_blend2.out
# CUDA_VISIBLE_DEVICES=1 nohup python train.py --data_path data_new/scene2_force1 data_new/scene2_force2 --expname scene2_nolossweight_2traj3 --configs arguments/dnerf/force.py --multi_cam --wandb > scene2_nolossweight_2traj3.out

# CUDA_VISIBLE_DEVICES=2 nohup python train.py --data_path data_new/scene2_force1 data_new/scene2_force2 data_new/scene2_force0 --n_train_cams 50 50 50 --n_test_cams 3 3 3 --expname scene2_3forcesl1 --configs arguments/dnerf/force.py --wandb > scene2_3forcesl1.out
# CUDA_VISIBLE_DEVICES=3 nohup python train.py --data_path data_new/scene2_force1 data_new/scene2_force2 data_new/scene2_force0 --n_train_cams 50 50 50 --n_test_cams 3 3 3 --expname scene2_3forcesl2 --configs arguments/dnerf/force.py --wandb > scene2_3forcesl2.out

# CUDA_VISIBLE_DEVICES=2 nohup python train.py --data_path data_new/scene2_force1 data_new/scene2_force2 data_new/scene2_force0 --n_train_cams 50 50 50 --n_test_cams 3 3 3 --expname scene2_3forcesl1 --configs arguments/dnerf/force.py --wandb > scene2_3forcesl1.out
# CUDA_VISIBLE_DEVICES=3 nohup python train.py --data_path data_new/scene2_force1 data_new/scene2_force2 data_new/scene2_force0 --n_train_cams 50 50 50 --n_test_cams 3 3 3 --expname scene2_3forcesl2 --configs arguments/dnerf/force.py --wandb > scene2_3forcesl2.out