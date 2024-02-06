# CUDA_VISIBLE_DEVICES=0 nohup python train.py --data_path data_new/scene2_force1 data_new/scene2_force2 --expname "scene2_lossweight1_2trajs5" --configs arguments/dnerf/force.py --multi_cam --wandb > scene2_lossweight1_2trajs5_correctexp.out &
# CUDA_VISIBLE_DEVICES=1 nohup python train.py --data_path data_new/scene2_force0 data_new/scene2_force1 data_new/scene2_force2 --expname "scene2_lossweight1_3trajs5" --configs arguments/dnerf/force.py --multi_cam --wandb > scene2_lossweight1_3trajs5_correctexp.out

# CUDA_VISIBLE_DEVICES=2 nohup python train.py --data_path data_new/scene2_force1 data_new/scene2_force2 --expname "scene2_nolossweigh_2traj" --configs arguments/dnerf/force.py --multi_cam --wandb > scene2_nolossweigh_2traj.out
# CUDA_VISIBLE_DEVICES=3 nohup python train.py --data_path data_new/scene2_force0 data_new/scene2_force1 data_new/scene2_force2 --expname "scene2_nolossweigh_3traj" --configs arguments/dnerf/force.py --multi_cam --wandb > scene2_nolossweigh_3traj.out

# CUDA_VISIBLE_DEVICES=1 nohup python train.py --data_path data_new/scene2_force1 data_new/scene2_force2 --expname scene2_nolosseigh_2traj3 --configs arguments/dnerf/force.py --multi_cam --wandb



# CUDA_VISIBLE_DEVICES=1 nohup python train.py --data_path data_new/scene2_force1 data_new/scene2_force2 --n_train_cams 30 50 --n_test_cams 2 2 --expname scene2_morecam2_l1 --configs arguments/dnerf/force.py --wandb > scene2_morecam2_l1.out &
# CUDA_VISIBLE_DEVICES=2 nohup python train.py --data_path data_new/scene2_force1 data_new/scene2_force2 --n_train_cams 30 50 --n_test_cams 2 2 --expname scene2_morecam2_l2 --configs arguments/dnerf/force.py --wandb --l2_plane_reg > scene2_morecam2_l2.out


CUDA_VISIBLE_DEVICES=0 nohup python train.py --data_path data_new/scene2_force1 data_new/scene2_force2 --n_train_cams 50 50 --n_test_cams 2 2 --expname scene2_l1andl2 --configs arguments/dnerf/force.py --wandb > scene2_l1andl2.out