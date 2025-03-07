_base_ = './dnerf_default.py'

USE_FORCE = True
USE_TIME = True
BLEND_TIME_FORCE = True
RESOLUTION = [64, 64, 64, 256]
INPUT_DIM = 4


OptimizationParams = dict(
    coarse_iterations = 10000, # 20000, # 10000, # default: 3000
    anchor_iterations = 0,
    densify_until_iter = 40000,
    iterations = 5000000,
    batch_size = 8,
    pruning_interval = 5000,
)

ModelParams = dict(
    extension='',
)

ModelHiddenParams = dict(
    kplanes_config = {
        'grid_dimensions': 2, # default: 2
        'input_coordinate_dim': INPUT_DIM,
        'output_coordinate_dim': 128, # 64, 128
        'resolution': RESOLUTION
    },
    defor_depth = 5,
    use_force = USE_FORCE,
    use_time = USE_TIME,
    blend_time_force = BLEND_TIME_FORCE,
    plane_tv_weight = 0.0001,
    force_weight = 0.01,
    time_smoothness_weight = 0.01, # default: 0.01
    l1_time_planes = 0.000, # 0001, # default: 0.0001
    l2_time_planes = 0.0005, # 0005,
    multires = [1, 2],
    no_dx=False,
    no_grid=False,
    no_ds=False,
    no_dr=False,
    no_do=True,
    no_dshs=False,
    force_pe=4,
    time_pe=4
)


# CUDA_VISIBLE_DEVICES=2 nohup python train.py --data_path data_new/scene2_force1 data_new/scene2_force2 --n_train_cams 50 50 --n_test_cams 2 2 --expname scene2_2force_5M_smalltimereg --configs arguments/dnerf/force.py --wandb > scene2_2force_5M_smalltimereg.out
# CUDA_VISIBLE_DEVICES=2 nohup python train.py --data_path data_new/scene2_force1 data_new/scene2_force2 data_new/scene2_force3 --n_train_cams 50 50 50 --n_test_cams 2 2 2 --expname scene2_batch8 --configs arguments/dnerf/force.py --wandb > scene2_batch8.out
# CUDA_VISIBLE_DEVICES=1 python train.py --data_path_train data_new/scene3redo --data_path_test data_new/scene3redo --n_train_cams 720 --n_test_cams 5 --expname pocfre_newforce_mask_multisteps9_nodsdshs2 --configs arguments/dnerf/force_blend.py --prev_frames 0 --start_checkpoint /home/baiting/4DGaussians/output/pocfre_newforce_mask_multisteps9/chkpnt_fine_269999.pth