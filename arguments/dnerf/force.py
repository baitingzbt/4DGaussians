_base_ = './dnerf_default.py'

USE_FORCE = True
USE_TIME = True
BLEND_TIME_FORCE = True
RESOLUTION = [64, 64, 64]
INPUT_DIM = 3

if USE_TIME:
    RESOLUTION += [64] # 64, 128
    INPUT_DIM += 1

if USE_FORCE and not BLEND_TIME_FORCE:
    RESOLUTION += [64] # [32, 32, 32, 64]
    INPUT_DIM += 1 # 4

OptimizationParams = dict(
    coarse_iterations = 3000, # default: 3000
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
    time_smoothness_weight = 0.1, # default: 0.01
    l1_time_planes =  0.001, # default: 0.0001
    multires = [1, 2]
)
# CUDA_VISIBLE_DEVICES=2 nohup python train.py --data_path data_new/scene2_force1 data_new/scene2_force2 --expname "force_scene2_blend2" --configs arguments/dnerf/force.py --multi_cam --wandb > force_scene2_blend2.out