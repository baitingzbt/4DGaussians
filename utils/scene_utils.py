import torch
import os
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif'] = ['Times New Roman']
import numpy as np
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from gaussian_renderer.renderer import render_for_image, render
from scene import Scene
import wandb
from typing import List, Union, Tuple, Dict
from .image_utils import psnr_np
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
from scene.dataset_readers import START_FRAME, MAX_FRAME
import pickle
FONT = ImageFont.truetype('./utils/TIMES.TTF', size=40) # 选择字体和字体大小
TXT_COLOR = (255, 0, 0)  # 白色
LABEL1_POS = (10, 10) # 选择标签的位置（左上角坐标

@torch.no_grad()
def render_training_image(
    scene: Scene,
    gaussians: GaussianModel,
    viewpoints: List[Camera],
    pipe,
    background,
    stage: str,
    iteration: int,
    time_now: float,
    save_video: bool = True,
    save_pointclound: bool = False,
    use_wandb: bool = False
) -> Tuple[float, float]:
    gaussians._deformation.eval()
    gaussians._deformation.deformation_net.eval()
    times = round(time_now / 60, 2)
    label2 = f"time: {times} mins"
    loss_and_psnr = []   # force_idx, cam_pose_idx, loss, psnr 
    def render_helper(i, viewpoint: Camera) -> Tuple[np.ndarray, float, float]:
        # if viewpoint.frame_step in [START_FRAME, MAX_FRAME-1]:
        #     return None, None, None
        image = render_for_image(viewpoint, gaussians, pipe, background, stage=stage)
        # image = render(
        #     [viewpoints[i-1], viewpoints[i], viewpoints[i+1]],
        #     gaussians, pipe, background, stage=stage
        # )[0]
        gt_np = viewpoint.original_image.permute(1, 2, 0).detach().cpu().numpy()
        image_np = image.permute(1, 2, 0).detach().cpu().numpy()  # 转换通道顺序为 (H, W, 3)
        loss = np.square(gt_np - image_np).mean()
        psnr = psnr_np(image_np, gt_np)
        label1 = f"stage:{stage},iter:{iteration}\nframe_t:{round(viewpoint.time, 2)}\nforce:{viewpoint.full_force}"
        image_np = np.concatenate((gt_np, image_np), axis=1)
        image_with_labels = Image.fromarray((np.clip(image_np, 0, 1) * 255).astype('uint8'))  # 转换为8位图像
        draw1 = ImageDraw.Draw(image_with_labels)
        label2_position = (image_with_labels.width - 100 - len(label2) * 10, 10)  # 右上角坐标
        draw1.text(LABEL1_POS, label1, fill=TXT_COLOR, font=FONT)
        draw1.text(label2_position, label2, fill=TXT_COLOR, font=FONT)
        image_with_label_arr = np.array(image_with_labels)
        loss_and_psnr.append(list(viewpoint.full_force) + [viewpoint.pose_idx, loss, psnr])
        return image_with_label_arr, loss, psnr
    

    render_base_path = os.path.join(scene.model_path, f"{stage}_render")
    point_cloud_path = os.path.join(render_base_path, "pointclouds")
    image_path = os.path.join(render_base_path, "images")
    os.makedirs(render_base_path, exist_ok=True)
    os.makedirs(point_cloud_path, exist_ok=True)
    os.makedirs(image_path, exist_ok=True)
    # image: (3, 800, 800)
    all_renders = []
    all_psnr = []
    all_test_loss = []
    for i, viewpoint in tqdm(enumerate(viewpoints), total=len(viewpoints)):
        # if viewpoint.frame_step in [0, 9]:
        #     continue
        img, loss, psnr = render_helper(i, viewpoint)
        if img is not None:
            all_renders.append(img)
            all_test_loss.append(loss)
            all_psnr.append(psnr)
    loss_and_psnr = np.array(loss_and_psnr)
    with open(stage + "_full_loss_and_psnr.pkl", "wb") as f:
        pickle.dump(loss_and_psnr, f)
    # breakpoint()
    # with ThreadPool() as pool:
    #     all_renders_infos: List = pool.map(
    #         render_helper,
    #         viewpoints
    #     )
    # all_renders = []
    # all_psnr = []
    # all_test_loss = []
    # for img, loss, psnr in all_renders_infos:
    #     if img and loss and psnr: # not One
    #         all_renders.append(img)
    #         all_test_loss.append(loss)
    #         all_psnr.append(psnr)

    avg_psnr = np.round(np.mean(np.array(all_psnr)), 2)
    avg_l1 = np.mean(np.array(all_test_loss))
    if save_video:
        video_path = os.path.join(image_path, f'{stage}_{iteration}.mp4')
        clip = ImageSequenceClip(all_renders, with_mask=False, fps=3)
        clip.write_videofile(video_path, logger=None)
        if use_wandb:
            wandb.log({f'video_{stage}': wandb.Video(video_path)})

    if save_pointclound:
        point_save_path = os.path.join(point_cloud_path, f"{iteration}.jpg")
        pc_mask = gaussians.get_opacity
        pc_mask = pc_mask > 0.1
        xyz = gaussians.get_xyz.detach()[pc_mask.squeeze()].cpu().permute(1, 0).numpy()
        visualize_and_save_point_cloud(xyz, viewpoints[0].R, viewpoints[0].T, point_save_path)
    
    gaussians._deformation.train()
    gaussians._deformation.deformation_net.train()
    return avg_l1, avg_psnr
    

def visualize_and_save_point_cloud(point_cloud, R, T, filename):
    # 创建3D散点图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    R = R.T
    # 应用旋转和平移变换
    T = -R.dot(T)
    transformed_point_cloud = np.dot(R, point_cloud) + T.reshape(-1, 1)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(transformed_point_cloud.T)  # 转置点云数据以匹配Open3D的格式
    # transformed_point_cloud[2,:] = -transformed_point_cloud[2,:]
    # 可视化点云
    ax.scatter(transformed_point_cloud[0], transformed_point_cloud[1], transformed_point_cloud[2], c='g', marker='o')
    ax.axis("off")
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')

    # 保存渲染结果为图片
    plt.savefig(filename)

