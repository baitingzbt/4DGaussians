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


# CUDA_VISIBLE_DEVICES=2 nohup python train.py --data_path data_new/scene2_force1 data_new/scene2_force2 data_new/scene2_force3 --n_train_cams 50 50 50 --n_test_cams 3 3 3 --expname scene2_3forces_nontrival --configs arguments/dnerf/force.py --wandb > scene2_3forces_nontrival.out &
# CUDA_VISIBLE_DEVICES=3 nohup python train.py --data_path data_new/scene2_force1 data_new/scene2_force2 data_new/scene2_force3 --n_train_cams 50 50 50 --n_test_cams 3 3 3 --expname scene2_3forces_nontrival_weighted --configs arguments/dnerf/force.py --wandb --weighted_loss > scene2_3forces_nontrival_weighted.out


# CUDA_VISIBLE_DEVICES=1 nohup python train.py --data_path data_new/scene2_force1 data_new/scene2_force2 --n_train_cams 50 50 --n_test_cams 2 2 --expname scene2_2force_dynamic_l2_debug --configs arguments/dnerf/force.py --weighted_loss 1 --wandb > scene2_2force_dynamic_l2.out &
# CUDA_VISIBLE_DEVICES=2 nohup python train.py --data_path data_new/scene2_force1 data_new/scene2_force2 data_new/scene2_force3 --n_train_cams 50 50 50 --n_test_cams 2 2 2 --expname scene2_3force_dynamic_l2_debug --configs arguments/dnerf/force.py --weighted_loss 1 --wandb > scene2_3force_dynamic_l2.out

# CUDA_VISIBLE_DEVICES=0 nohup python train.py --data_path data_new/scene2_force1 data_new/scene2_force2 data_new/scene2_force3 --n_train_cams 50 50 50 --n_test_cams 2 2 2 --expname scene2_3force_momentum_batch1_l1 --configs arguments/dnerf/force.py --wandb > scene2_3force_momentum_batch1_l1.out
# CUDA_VISIBLE_DEVICES=1 nohup python train.py --data_path data_new/scene2_force1 data_new/scene2_force2 data_new/scene2_force3 --n_train_cams 50 50 50 --n_test_cams 2 2 2 --expname scene2_3force_momentum_batch8 --configs arguments/dnerf/force.py --wandb > scene2_3force_momentum_batch8.out
# CUDA_VISIBLE_DEVICES=0 nohup python train.py --data_path data_new/scene2_force1 data_new/scene2_force2 data_new/scene2_force3 --n_train_cams 50 50 50 --n_test_cams 2 2 2 --expname scene2_3force_batch8_momen2 --configs arguments/dnerf/force.py --wandb > scene2_3force_batch8_momen2.out


# CUDA_VISIBLE_DEVICES=3 nohup python train.py \
#     --data_path_train data_new/scene2_force3_250 data_new/scene2_force3_300 data_new/scene2_force3_350 data_new/scene2_force3_380 data_new/scene2_force3_400 data_new/scene2_force3_450 \
#     --data_path_test data_new/scene2_force3_275 data_new/scene2_force3_325 data_new/scene2_force3_375 data_new/scene2_force3_390 data_new/scene2_force3_425 \
#     --n_train_cams 50 50 50 50 50 50 \
#     --n_test_cams 2 2 2 2 2 \
#     --expname scene2_novel_intensity5 \
#     --configs arguments/dnerf/force.py \
#     --wandb > scene2_novel_intensity5.out \
# &
# CUDA_VISIBLE_DEVICES=2 nohup python train.py \
#     --data_path_train data_new/scene2_force3_250 data_new/scene2_force3_300 data_new/scene2_force3_350 data_new/scene2_force3_380 data_new/scene2_force3_400 \
#     --data_path_test data_new/scene2_force3_275 data_new/scene2_force3_325 data_new/scene2_force3_375 data_new/scene2_force3_390 \
#     --n_train_cams 50 50 50 50 50 \
#     --n_test_cams 2 2 2 2 \
#     --expname scene2_novel_intensity6 \
#     --configs arguments/dnerf/force.py \
#     --wandb > scene2_novel_intensity6.out \
# &
# CUDA_VISIBLE_DEVICES=1 nohup python train.py \
#     --data_path_train data_new/scene2_force3_250 data_new/scene2_force3_300 data_new/scene2_force3_350 data_new/scene2_force3_380 data_new/scene2_force3_400 data_new/scene2_force3_450 \
#     --data_path_test data_new/scene2_force3_275 data_new/scene2_force3_325 data_new/scene2_force3_375 data_new/scene2_force3_390 data_new/scene2_force3_425 \
#     --n_train_cams 50 50 50 50 50 50 \
#     --n_test_cams 2 2 2 2 2 \
#     --expname scene2_novel_intensity5_less_coarse \
#     --configs arguments/dnerf/coarse1k.py \
#     --wandb > scene2_novel_intensity5_less_coarse.out \
# &
# CUDA_VISIBLE_DEVICES=0 nohup python train.py \
#     --data_path_train data_new/scene2_force3_250 data_new/scene2_force3_300 data_new/scene2_force3_350 data_new/scene2_force3_380 data_new/scene2_force3_400 \
#     --data_path_test data_new/scene2_force3_275 data_new/scene2_force3_325 data_new/scene2_force3_375 data_new/scene2_force3_390 \
#     --n_train_cams 50 50 50 50 50 \
#     --n_test_cams 2 2 2 2 \
#     --expname scene2_novel_intensity6_less_coarse \
#     --configs arguments/dnerf/coarse1k.py \
#     --wandb > scene2_novel_intensity6_less_coarse.out



# CUDA_VISIBLE_DEVICES=2 nohup python train.py \
#     --data_path_train data_new/scene2_force3_250 data_new/scene2_force3_300 data_new/scene2_force3_350 data_new/scene2_force3_380 \
#     --data_path_test data_new/scene2_force3_275 data_new/scene2_force3_325 data_new/scene2_force3_375 data_new/scene2_force3_390 \
#     --n_train_cams 50 50 50 50 \
#     --n_test_cams 2 2 2 2 \
#     --expname scene2_novel_intensity8_cont \
#     --configs arguments/dnerf/force_blend.py \
#     --start_checkpoint /home/baiting/4DGaussians/output/scene2_novel_intensity8/chkpnt_fine_99999.pth \
#     --wandb > scene2_novel_intensity8_cont.out

# CUDA_VISIBLE_DEVICES=2 nohup python train.py \
#     --data_path_train data_new/scene2_force3_250 data_new/scene2_force3_300 data_new/scene2_force3_350 data_new/scene2_force3_380 \
#     --data_path_test data_new/scene2_force3_275 data_new/scene2_force3_325 data_new/scene2_force3_375 data_new/scene2_force3_390 \
#     --n_train_cams 50 50 50 50 \
#     --n_test_cams 2 2 2 2 \
#     --expname scene2_opacity_yes \
#     --configs arguments/dnerf/force_blend.py \
#     --wandb > scene2_opacity_yes.out \
# &
# CUDA_VISIBLE_DEVICES=3 nohup python train.py \
#     --data_path_train \
#         data_new/scene2_dir_train1 data_new/scene2_dir_train2 data_new/scene2_dir_train3 data_new/scene2_dir_train4 \
#         data_new/scene2_dir_train5 data_new/scene2_dir_train6 data_new/scene2_dir_train7 data_new/scene2_dir_train8 \
#     --data_path_test \
#         data_new/scene2_dir_test1 data_new/scene2_dir_test2 data_new/scene2_dir_test3 data_new/scene2_dir_test4 \
#     --n_train_cams 50 50 50 50 50 50 50 50 \
#     --n_test_cams 3 3 3 3 \
#     --expname scene2_new_dir1_opacity_yes \
#     --configs arguments/dnerf/force_blend.py \
#     --wandb > scene2_new_dir1_opacity_yes.out


# CUDA_VISIBLE_DEVICES=3 nohup python train.py \
#     --data_path_train \
#         data_new/scene2_dir_train1 data_new/scene2_dir_train2 data_new/scene2_dir_train5 data_new/scene2_dir_train6 \
#     --data_path_test \
#         data_new/scene2_dir_test2 data_new/scene2_dir_test3 data_new/scene2_dir_test5 data_new/scene2_dir_test6 \
#     --n_train_cams 50 50 50 50 \
#     --n_test_cams 3 3 3 3 \
#     --expname scene2_new_dir1_less_yes_controlledreg \
#     --configs arguments/dnerf/force_blend.py \
#     --wandb > scene2_new_dir1_less_yes_controlledreg.out


# CUDA_VISIBLE_DEVICES=2 nohup python train.py \
#     --data_path_train data_new/scene2_force3_250 data_new/scene2_force3_300 data_new/scene2_force3_350 data_new/scene2_force3_380 \
#     --data_path_test data_new/scene2_force3_275 data_new/scene2_force3_325 data_new/scene2_force3_375 data_new/scene2_force3_390 \
#     --n_train_cams 50 50 50 50 \
#     --n_test_cams 3 3 3 3 \
#     --expname scene2_novel_intensity_anchor2 \
#     --configs arguments/dnerf/force_blend_anchor.py \
#     --wandb > scene2_novel_intensity_anchor2.out \
# &
# CUDA_VISIBLE_DEVICES=1 nohup python train.py \
#     --data_path_train data_new/scene2_force3_250 data_new/scene2_force3_300 data_new/scene2_force3_350 data_new/scene2_force3_380 \
#     --data_path_test data_new/scene2_force3_275 data_new/scene2_force3_325 data_new/scene2_force3_375 data_new/scene2_force3_390 \
#     --n_train_cams 50 50 50 50 \
#     --n_test_cams 3 3 3 3 \
#     --expname scene2_novel_intensity_35_lesspts \
#     --configs arguments/dnerf/force_blend.py \
#     --wandb > scene2_novel_intensity_35_lesspts.out


# CUDA_VISIBLE_DEVICES=1 nohup python train.py \
#     --data_path_train data_new/scene2_force3_250 data_new/scene2_force3_300 data_new/scene2_force3_350 data_new/scene2_force3_380 \
#     --data_path_test data_new/scene2_force3_275 data_new/scene2_force3_325 data_new/scene2_force3_375 data_new/scene2_force3_390 \
#     --n_train_cams 50 50 50 50 \
#     --n_test_cams 3 3 3 3 \
#     --expname scene2_novel_intensity_9anchors_force_nocast \
#     --configs arguments/dnerf/force_blend_anchor.py \
#     --wandb > scene2_novel_intensity_9anchors_force_nocast.out

# CUDA_VISIBLE_DEVICES=3 nohup python train.py \
#     --data_path_train data_new/scene2_force3_250 data_new/scene2_force3_300 data_new/scene2_force3_350 data_new/scene2_force3_380 \
#     --data_path_test data_new/scene2_force3_275 data_new/scene2_force3_325 data_new/scene2_force3_375 data_new/scene2_force3_390 \
#     --n_train_cams 50 50 50 50 \
#     --n_test_cams 3 3 3 3 \
#     --expname scene2_novel_intensity_9anchors_force_cast \
#     --configs arguments/dnerf/force_blend_anchor.py \
#     --wandb > scene2_novel_intensity_9anchors_force_cast.out

# CUDA_VISIBLE_DEVICES=3 nohup python train.py --data_path_train data_new/scene2_force1 --data_path_test data_new/scene2_force1 --n_train_cams 50 --n_test_cams 4 --expname scene2_test_jumpframes --configs arguments/dnerf/force_blend.py --wandb > scene2_test_jumpframes.out 


# CUDA_VISIBLE_DEVICES=3 nohup python train.py \
#     --data_path_train data_new/scene2_force3_250 data_new/scene2_force3_300 data_new/scene2_force3_350 data_new/scene2_force3_380 \
#     --data_path_test data_new/scene2_force3_275 data_new/scene2_force3_325 data_new/scene2_force3_375 data_new/scene2_force3_390 \
#     --n_train_cams 50 50 50 50 \
#     --n_test_cams 3 3 3 3 \
#     --expname scene2_novel_intensity_phases \
#     --configs arguments/dnerf/force_blend.py \
#     --wandb > scene2_novel_intensity_phases.out

CUDA_VISIBLE_DEVICES=2 nohup python train.py \
    --data_path_train data_new/scene2_dir_train1 data_new/scene2_dir_train2 data_new/scene2_dir_train5 data_new/scene2_dir_train6 \
    --data_path_test data_new/scene2_dir_test2 data_new/scene2_dir_test3 data_new/scene2_dir_test5 data_new/scene2_dir_test6 \
    --n_train_cams 50 50 50 50 \
    --n_test_cams 3 3 3 3 \
    --expname scene2_directions_anchors_cast \
    --configs arguments/dnerf/force_blend_anchor.py \
    --wandb --phases > scene2_directions_anchors_cast.out \
&
CUDA_VISIBLE_DEVICES=1 nohup python train.py \
    --data_path_train data_new/scene2_dir_train1 data_new/scene2_dir_train2 data_new/scene2_dir_train5 data_new/scene2_dir_train6 \
    --data_path_test data_new/scene2_dir_test2 data_new/scene2_dir_test3 data_new/scene2_dir_test5 data_new/scene2_dir_test6 \
    --n_train_cams 50 50 50 50 \
    --n_test_cams 3 3 3 3 \
    --expname scene2_directions_phases \
    --configs arguments/dnerf/force_blend_anchor.py \
    --wandb --anchors > scene2_directions_phases.out

CUDA_VISIBLE_DEVICES=1 nohup python train.py \
    --data_path_train data_new/scene2_dir_train1 data_new/scene2_dir_train2 data_new/scene2_dir_train5 data_new/scene2_dir_train6 \
    --data_path_test data_new/scene2_dir_test2 data_new/scene2_dir_test3 data_new/scene2_dir_test5 data_new/scene2_dir_test6 \
    --n_train_cams 50 50 50 50 \
    --n_test_cams 3 3 3 3 \
    --expname scene2_directions_anchors_cast2 \
    --configs arguments/dnerf/force_blend_anchor.py \
    --wandb --anchors > scene2_directions_anchors_cast2.out

CUDA_VISIBLE_DEVICES=1 nohup python train.py \
    --data_drive /sdd/baiting/4DGaussians \
    --data_path_train data_new/scene2_dir_train1 data_new/scene2_dir_train2 data_new/scene2_dir_train5 data_new/scene2_dir_train6 \
    --data_path_test data_new/scene2_dir_test2 data_new/scene2_dir_test3 data_new/scene2_dir_test5 data_new/scene2_dir_test6 \
    --n_train_cams 50 50 50 50 \
    --n_test_cams 3 3 3 3 \
    --expname scene2_directions_recurssive \
    --configs arguments/dnerf/force_blend.py \
    --wandb > scene2_directions_recurssive.out

CUDA_VISIBLE_DEVICES=0 nohup python train.py \
    --data_drive /sdd/baiting/4DGaussians \
    --data_path_train data_new/scene2_force3_250 data_new/scene2_force3_300 data_new/scene2_force3_350 data_new/scene2_force3_380 \
    --data_path_test data_new/scene2_force3_275 data_new/scene2_force3_325 data_new/scene2_force3_375 data_new/scene2_force3_390 \
    --n_train_cams 50 50 50 50 \
    --n_test_cams 3 3 3 3 \
    --expname scene2_intensity_uniforminit \
    --configs arguments/dnerf/force_blend.py \
    --wandb > scene2_intensity_uniforminit.out




# CUDA_VISIBLE_DEVICES=1 nohup python train.py \
#     --data_drive /sdd/baiting/4DGaussians \
#     --data_path_train \
#         data_new/scene2_dir_train1 data_new/scene2_dir_train2 data_new/scene2_dir_train3 data_new/scene2_dir_train4 \
#         data_new/scene2_dir_train5 data_new/scene2_dir_train6 data_new/scene2_dir_train7 data_new/scene2_dir_train8 \
#     --data_path_test \
#         data_new/scene2_dir_test1 data_new/scene2_dir_test2 data_new/scene2_dir_test3 data_new/scene2_dir_test4 \
#         data_new/scene2_dir_test5 data_new/scene2_dir_test6 \
#     --n_train_cams 50 50 50 50 50 50 50 50\
#     --n_test_cams 3 3 3 3 3 3\
#     --expname scene2_directions_full_cont2 \
#     --configs arguments/dnerf/force_blend.py \
#     --start_checkpoint /sdd/baiting/4DGaussians/output/scene2_directions_full_cont/chkpnt_fine_199999.pth \
#     --wandb > scene2_directions_full_cont2.out

# CUDA_VISIBLE_DEVICES=1 nohup python train.py \
#     --data_drive /sdd/baiting/4DGaussians \
#     --data_path_train \
#         data_new/scene2_dir_train1 data_new/scene2_dir_train2 data_new/scene2_dir_train3 data_new/scene2_dir_train4 \
#         data_new/scene2_dir_train5 data_new/scene2_dir_train6 data_new/scene2_dir_train7 data_new/scene2_dir_train8 \
#     --data_path_test \
#         data_new/scene2_dir_test1 data_new/scene2_dir_test2 data_new/scene2_dir_test3 data_new/scene2_dir_test4 \
#         data_new/scene2_dir_test5 data_new/scene2_dir_test6 \
#     --n_train_cams 50 50 50 50 50 50 50 50\
#     --n_test_cams 3 3 3 3 3 3 \
#     --expname dirs8_uniform_sep_noforceexp \
#     --configs arguments/dnerf/force_sep.py \
#     --wandb > dirs8_uniform_sep_noforceexp.out

CUDA_VISIBLE_DEVICES=1 nohup python train.py \
    --data_drive /sdd/baiting/4DGaussians \
    --data_path_train \
        data_new/scene2_dir_train1 data_new/scene2_dir_train2 data_new/scene2_dir_train3 data_new/scene2_dir_train4 \
        data_new/scene2_dir_train5 data_new/scene2_dir_train6 data_new/scene2_dir_train7 data_new/scene2_dir_train8 \
        data_new/scene2_dir_train9 data_new/scene2_dir_train10 data_new/scene2_dir_train11 data_new/scene2_dir_train12 \
        data_new/scene2_dir_train13 data_new/scene2_dir_train14 data_new/scene2_dir_train15 data_new/scene2_dir_train16 \
    --data_path_test \
        data_new/scene2_dir_test1 data_new/scene2_dir_test2 data_new/scene2_dir_test3 data_new/scene2_dir_test4 \
        data_new/scene2_dir_test5 data_new/scene2_dir_test6 \
    --n_train_cams 30 30 30 30 30 30 30 30 30 30 30 30 30 30 30 30\
    --n_test_cams 3 3 3 3 3 3 \
    --expname dirs16_blend_knn \
    --configs arguments/dnerf/force_blend.py \
    --wandb > dirs16_blend_knn.out

CUDA_VISIBLE_DEVICES=1 nohup python train.py \
    --data_drive . \
    --data_path_train \
        data_new/scene2_dir_train1 data_new/scene2_dir_train2 data_new/scene2_dir_train3 data_new/scene2_dir_train4 \
        data_new/scene2_dir_train5 data_new/scene2_dir_train6 data_new/scene2_dir_train7 data_new/scene2_dir_train8 \
        data_new/scene2_dir_train9 data_new/scene2_dir_train10 data_new/scene2_dir_train11 data_new/scene2_dir_train12 \
        data_new/scene2_dir_train13 data_new/scene2_dir_train14 data_new/scene2_dir_train15 data_new/scene2_dir_train16 \
    --data_path_test \
        data_new/scene2_dir_test1 data_new/scene2_dir_test2 data_new/scene2_dir_test3 data_new/scene2_dir_test4 \
        data_new/scene2_dir_test5 data_new/scene2_dir_test6 \
    --n_train_cams 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10\
    --n_test_cams 3 3 3 3 3 3 \
    --expname dirs16_sep_knn2 \
    --configs arguments/dnerf/force_sep.py \
    --wandb > dirs16_sep_knn2.out


CUDA_VISIBLE_DEVICES=2 nohup python train.py \
    --data_drive /sdd/baiting/4DGaussians \
    --data_path_train data_new/scene2_force3_250 data_new/scene2_force3_300 data_new/scene2_force3_350 data_new/scene2_force3_380 \
    --data_path_test data_new/scene2_force3_275 data_new/scene2_force3_325 data_new/scene2_force3_375 data_new/scene2_force3_390 \
    --n_train_cams 50 50 50 50 \
    --n_test_cams 3 3 3 3 \
    --expname inten4_uniform_blend_forceexp_knn_angle \
    --configs arguments/dnerf/force_blend.py \
    --wandb > inten4_uniform_blend_forceexp_knn_angle.out

# CUDA_VISIBLE_DEVICES=0 python train.py --data_drive ../ --data_path_train data_new/scene2_dir_train1 data_new/scene2_dir_train2 data_new/scene2_dir_train3 --data_path_test data_new/scene2_dir_test1 data_new/scene2_dir_test2 data_new/scene2_dir_test3 --n_train_cams 5 5 5 --n_test_cams 5 5 5 --expname test_read --configs arguments/dnerf/less_coarse.py
# CUDA_VISIBLE_DEVICES=0 python train.py --data_drive /sdd/baiting/4DGaussians --data_path_train data_new/scene2_force1 --data_path_test data_new/scene2_force1 --n_train_cams 5 --n_test_cams 2 --expname scene2_test_forceplane --configs arguments/dnerf/less_coarse.py