_base_ = './dnerf_default.py'

USE_FORCE = True
USE_TIME = False
BLEND_TIME_FORCE = False
RESOLUTION = [64, 64, 64]
INPUT_DIM = 3

if USE_TIME and USE_FORCE:
    # add 1 for time
    RESOLUTION += [128]
    INPUT_DIM += 1
    # add 1 for force, if not blending
    if not BLEND_TIME_FORCE:
        RESOLUTION += [128]
        INPUT_DIM += 1
else:
    # only using 1 of time / force, using none is invalid
    assert USE_TIME or USE_FORCE
    # shouldn't blend
    assert not BLEND_TIME_FORCE
    # add dimension for that
    RESOLUTION += [128]
    INPUT_DIM += 1

OptimizationParams = dict(
    coarse_iterations = 2000, # 10000, # default: 3000
    anchor_iterations = 0,
    densify_until_iter = 10000,
    iterations = 5000000,
    batch_size = 128,
    pruning_interval = 100, # don't prune
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
    # bounds=2.0,
    # posebase_pe = 2,
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
    no_ds=True,
    no_dr=True,
    no_do=True,
    no_dshs=True,
)


# CUDA_VISIBLE_DEVICES=2 nohup python train.py --data_path data_new/scene2_force1 data_new/scene2_force2 --n_train_cams 50 50 --n_test_cams 2 2 --expname scene2_2force_5M_smalltimereg --configs arguments/dnerf/force.py --wandb > scene2_2force_5M_smalltimereg.out
# CUDA_VISIBLE_DEVICES=2 nohup python train.py --data_path data_new/scene2_force1 data_new/scene2_force2 data_new/scene2_force3 --n_train_cams 50 50 50 --n_test_cams 2 2 2 --expname scene2_batch8 --configs arguments/dnerf/force.py --wandb > scene2_batch8.out
# CUDA_VISIBLE_DEVICES=1 python train.py --data_path_train data_new/scene3redo --data_path_test data_new/scene3redo --n_train_cams 720 --n_test_cams 5 --expname pocfre_newforce_mask_multisteps9_nodsdshs2 --configs arguments/dnerf/force_blend.py --prev_frames 0 --start_checkpoint /home/baiting/4DGaussians/output/pocfre_newforce_mask_multisteps9/chkpnt_fine_269999.pth