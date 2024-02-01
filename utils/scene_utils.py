import torch
import os
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif'] = ['Times New Roman']
import numpy as np
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from gaussian_renderer.renderer import render
import wandb
from typing import List

@torch.no_grad()
def render_training_image(
    scene,
    gaussians: GaussianModel,
    viewpoints: List[Camera],
    pipe,
    background,
    stage: str,
    iteration,
    time_now,
    save_video: bool = True,
    save_pointclound: bool = True,
    save_images: bool = False,
    use_wandb: bool = False
) -> None:
    gaussians._deformation.eval()
    gaussians._deformation.deformation_net.eval()
    def render_helper(viewpoint: Camera, path: str = None) -> np.ndarray:
        # if stage.startswith('coarse') or viewpoint.time == 0.0:
        #     prev_hidden = None
        image, _, _, _, depth = render(viewpoint, gaussians, pipe, background, stage=stage)
        times =  time_now / 60
        label2 = "time:%.2f" % times + 'mins'
        label1 = f"stage:{stage},iter:{iteration}\nframe_t:{viewpoint.time}"
        gt_np = viewpoint.original_image.permute(1, 2, 0).cpu().numpy()
        image_np = image.permute(1, 2, 0).cpu().numpy()  # 转换通道顺序为 (H, W, 3)
        depth_np = depth.permute(1, 2, 0).cpu().numpy()
        depth_np /= depth_np.max()
        depth_np = np.repeat(depth_np, 3, axis=2)
        image_np = np.concatenate((gt_np, image_np, depth_np), axis=1)
        image_with_labels = Image.fromarray((np.clip(image_np, 0, 1) * 255).astype('uint8'))  # 转换为8位图像
        # 创建PIL图像对象的副本以绘制标签
        draw1 = ImageDraw.Draw(image_with_labels)
        # 选择字体和字体大小
        font = ImageFont.truetype('./utils/TIMES.TTF', size=40)  # 请将路径替换为您选择的字体文件路径
        # 选择文本颜色
        text_color = (255, 0, 0)  # 白色
        # 选择标签的位置（左上角坐标）
        label1_position = (10, 10)
        label2_position = (image_with_labels.width - 100 - len(label2) * 10, 10)  # 右上角坐标
        # 在图像上添加标签
        draw1.text(label1_position, label1, fill=text_color, font=font)
        draw1.text(label2_position, label2, fill=text_color, font=font)

        if save_images:
            image_with_labels.save(path)
        image_with_label_arr = np.array(image_with_labels)
        return image_with_label_arr # , hidden
    
    render_base_path = os.path.join(scene.model_path, f"{stage}_render")
    point_cloud_path = os.path.join(render_base_path,"pointclouds")
    image_path = os.path.join(render_base_path,"images")
    os.makedirs(render_base_path, exist_ok=True)
    os.makedirs(point_cloud_path, exist_ok=True)
    os.makedirs(image_path, exist_ok=True)
    # image:3,800,800
    
    point_save_path = os.path.join(point_cloud_path,f"{iteration}.jpg")
    all_renders = []
    for idx, viewpoint in enumerate(viewpoints):
        image_save_path = os.path.join(image_path,f"{iteration}_{idx}.png") if save_images else None
        image_with_label_arr = render_helper(viewpoint, image_save_path)
        all_renders.append(image_with_label_arr)
    
    if save_video:
        video_path = os.path.join(image_path, f'{iteration}.mp4')
        clip = ImageSequenceClip(all_renders, with_mask=False, fps=30)
        clip.write_videofile(video_path, logger=None)
        if use_wandb:
            wandb.log({f'video_{stage}': wandb.Video(video_path)})

    if save_pointclound:
        pc_mask = gaussians.get_opacity
        pc_mask = pc_mask > 0.1
        xyz = gaussians.get_xyz.detach()[pc_mask.squeeze()].cpu().permute(1,0).numpy()
        visualize_and_save_point_cloud(xyz, viewpoint.R, viewpoint.T, point_save_path)
    
    gaussians._deformation.train()
    gaussians._deformation.deformation_net.train()
    

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

