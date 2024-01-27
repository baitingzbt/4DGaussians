_base_ = './dnerf_default.py'

USE_FORCE = True
USE_TIME = True
RESOLUTION = [64, 64, 64]
INPUT_DIM = 3

if USE_TIME:
    RESOLUTION += [64] # 64, 128
    INPUT_DIM += 1

if USE_FORCE:
    RESOLUTION += [64] # [32, 32, 32, 64]
    INPUT_DIM += 1 # 4

OptimizationParams = dict(
    coarse_iterations = 6000, # default: 3000
    iterations = 300000
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
    multires = [1, 2]
)