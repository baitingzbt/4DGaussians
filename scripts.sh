CUDA_VISIBLE_DEVICES=2 nohup python train.py --data_path data_new/scene2_force1 data_new/scene2_force2 --expname "scene2_lossweight1_2trajs" --configs arguments/dnerf/force.py --multi_cam --wandb > scene2_lossweight1_2trajs2.out &
CUDA_VISIBLE_DEVICES=3 nohup python train.py --data_path data_new/scene2_force0 data_new/scene2_force1 data_new/scene2_force2 --expname "scene2_lossweight1_3trajs" --configs arguments/dnerf/force.py --multi_cam --wandb > scene2_lossweight1_3trajs2.out

