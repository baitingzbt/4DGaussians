# CUDA_VISIBLE_DEVICES=0 nohup python train.py --data_path data_new/scene2_force1 data_new/scene2_force2 --expname "scene2_lossweight1_2trajs5" --configs arguments/dnerf/force.py --multi_cam --wandb > scene2_lossweight1_2trajs5_correctexp.out &
# CUDA_VISIBLE_DEVICES=1 nohup python train.py --data_path data_new/scene2_force0 data_new/scene2_force1 data_new/scene2_force2 --expname "scene2_lossweight1_3trajs5" --configs arguments/dnerf/force.py --multi_cam --wandb > scene2_lossweight1_3trajs5_correctexp.out

# CUDA_VISIBLE_DEVICES=2 nohup python train.py --data_path data_new/scene2_force1 data_new/scene2_force2 --expname "scene2_nolossweigh_2traj" --configs arguments/dnerf/force.py --multi_cam --wandb > scene2_nolossweigh_2traj.out
# CUDA_VISIBLE_DEVICES=3 nohup python train.py --data_path data_new/scene2_force0 data_new/scene2_force1 data_new/scene2_force2 --expname "scene2_nolossweigh_3traj" --configs arguments/dnerf/force.py --multi_cam --wandb > scene2_nolossweigh_3traj.out

# CUDA_VISIBLE_DEVICES=1 nohup python train.py --data_path data_new/scene2_force1 data_new/scene2_force2 --expname scene2_nolosseigh_2traj3 --configs arguments/dnerf/force.py --multi_cam --wandb



# CUDA_VISIBLE_DEVICES=1 nohup python train.py --data_path data_new/scene2_force1 data_new/scene2_force2 --n_train_cams 30 50 --n_test_cams 2 2 --expname scene2_morecam2_l1 --configs arguments/dnerf/force.py --wandb > scene2_morecam2_l1.out &
# CUDA_VISIBLE_DEVICES=2 nohup python train.py --data_path data_new/scene2_force1 data_new/scene2_force2 --n_train_cams 30 50 --n_test_cams 2 2 --expname scene2_morecam2_l2 --configs arguments/dnerf/force.py --wandb --l2_plane_reg > scene2_morecam2_l2.out


# CUDA_VISIBLE_DEVICES=3 nohup python train.py --data_path data_new/scene2_force1 data_new/scene2_force2 --n_train_cams 50 50 --n_test_cams 2 2 --expname scene2_l1andl2 --configs arguments/dnerf/force.py --wandb > scene2_l1andl2.out
# CUDA_VISIBLE_DEVICES=2 nohup python train.py --data_path data_new/scene2_force1 data_new/scene2_force2 --n_train_cams 50 50 --n_test_cams 2 2 --expname scene2_l1andl2_2 --configs arguments/dnerf/force.py --wandb > scene2_l1andl2_2.out
# CUDA_VISIBLE_DEVICES=1 nohup python train.py --data_path data_new/scene2_force1 data_new/scene2_force2 --n_train_cams 50 50 --n_test_cams 3 3 --expname scene2_l1l2_decaytime --configs arguments/dnerf/force.py --wandb > scene2_l1l2_decaytime.out

# CUDA_VISIBLE_DEVICES=0 nohup python train.py --data_path data_new/scene2_force1 data_new/scene2_force2 --n_train_cams 50 50 --n_test_cams 3 3 --expname scene2_lossweight_2force --configs arguments/dnerf/force.py --weighted_loss --wandb > scene2_lossweight_2force.out &
# CUDA_VISIBLE_DEVICES=1 nohup python train.py --data_path data_new/scene2_force0 data_new/scene2_force1 data_new/scene2_force2 --n_train_cams 50 50 50 --n_test_cams 3 3 3 --expname scene2_lossweight_3force --configs arguments/dnerf/force.py --weighted_loss --wandb > scene2_lossweight_3force.out


CUDA_VISIBLE_DEVICES=2 nohup python train.py --data_path data_new/scene2_force1 data_new/scene2_force2 data_new/scene2_force3 --n_train_cams 50 50 50 --n_test_cams 3 3 3 --expname scene2_3forces_nontrival --configs arguments/dnerf/force.py --wandb > scene2_3forces_nontrival.out
CUDA_VISIBLE_DEVICES=3 nohup python train.py --data_path data_new/scene2_force1 data_new/scene2_force2 data_new/scene2_force3 --n_train_cams 50 50 50 --n_test_cams 3 3 3 --expname scene2_3forces_nontrival_weighted --configs arguments/dnerf/force.py --wandb --weighted_loss > scene2_3forces_nontrival_weighted.out