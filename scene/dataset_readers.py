#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from scene.cameras import Camera
from typing import List, Set, Any, Optional, Dict, NamedTuple, Tuple
from multiprocessing.pool import ThreadPool
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import torch
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from utils.general_utils import PILtoTorch
from tqdm import tqdm
import re
import time
MAX_FRAME = 25
START_FRAME = 0 # 0: use all frames, otherwise drops first n frames
CAM_H, CAM_W = 480, 480

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    time : float
    mask: np.array
    force: np.array = None
    force_idx: int = -1
   
class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: List[Camera]
    test_cameras: List[Camera]
    nerf_normalization: dict
    ply_path: str
    maxtime: int

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal
    cam_centers = []
    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])
    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1
    translate = -center
    return {"translate": translate, "radius": radius}

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'f4'), ('green', 'f4'), ('blue', 'f4')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    # breakpoint()
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


# def readCamerasFromTransforms(
#     path: str,
#     transformsfile: str,
#     white_background: bool,
#     extension: str,
#     mapper: Dict,
#     force_idx: int,
# ) -> List[CameraInfo]:
#     cam_infos = []
#     with open(os.path.join(path, transformsfile)) as json_file:
#         contents: Dict = json.load(json_file)
#     fovx = contents['camera_angle_x'] if 'camera_angle_x' in contents.keys() \
#         else focal2fov(contents['fl_x'],contents['w'])
#     for idx, frame in enumerate(contents["frames"]):
#         if idx >= MAX_FRAME or idx < START_FRAME:
#             break
#         cam_name = os.path.join(path, frame["file_path"] + extension)
#         time = mapper[frame["time"]]
#         matrix = np.linalg.inv(np.array(frame["transform_matrix"]))
#         R = -np.transpose(matrix[:3, :3])
#         R[:, 0] = -R[:, 0]
#         T = -matrix[:3, 3]

#         image_path = os.path.join(path, cam_name)
#         image_name = Path(cam_name).stem
#         image = Image.open(image_path)
#         im_data = np.array(image.convert("RGBA"))
#         bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])
#         norm_data = im_data / 255.0
#         arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
#         image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")
#         image = PILtoTorch(image, (CAM_W, CAM_H))
#         fovy = focal2fov(fov2focal(fovx, image.shape[1]), image.shape[2])
#         force = np.array(frame['force']); assert force.shape == (7, )
#         cam_infos.append(
#             CameraInfo(
#                 uid=idx, R=R, T=T, FovY=fovy, FovX=fovx, image=image, image_path=image_path,
#                 image_name=image_name, width=image.shape[1], height=image.shape[2], time = time, mask=None, force=force, force_idx=force_idx
#             )
#         )
#     return cam_infos

# # use with cautious, lots of special case and magic numbers to maximize reading speed
# def readCamerasFromShortTransforms(
#     path: str,
#     transformsfile: str,
#     mapper: Dict,
#     force_idx: int,
# ) -> List[Camera]:
#     cams = []
#     with open(f"{path}/{transformsfile[:-5]}_short.json") as json_file:
#         contents: Dict = json.load(json_file)
#     matrix = -np.linalg.inv(np.array(contents["transform_matrix"]))
#     R = np.transpose(matrix[:3, :3])
#     R[:, 0] = -R[:, 0]
#     # H, W = contents['height'], contents['width']    # original H, W
#     # NEW_H, NEW_W = 600, 600    # target H, W
#     force = np.array(contents['force'])[3:]  #  directly drop positions here
#     force /= np.array([1, 1, 1, 1000])
#     for idx, frame in enumerate(contents["frames"]):
#         if idx >= MAX_FRAME:
#             break
#         image = Image.open(os.path.join(path, frame["file_path"])).resize((CAM_W, CAM_H))
#         norm_data =  np.array(image.convert("RGBA")) / 255.0
#         # assume white_background = True
#         arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + (1 - norm_data[:, :, 3:4])
#         cams.append(
#             Camera(
#                 R=R,
#                 T=matrix[:3, 3],
#                 FoVx=contents['camera_angle_x'], # assume H and W are the same to save compute
#                 FoVy=contents['camera_angle_x'], # assume H and W are the same to save compute
#                 image=torch.tensor(arr, dtype=torch.float32).permute(2, 0, 1),
#                 time=mapper[frame["time"]],
#                 frame_step=idx,
#                 force=force,
#                 force_idx=force_idx
#             )
#         )
#     return cams


import math
def force_process(force: np.ndarray) -> float:
    # Calculate the angle in radians using atan2, which takes into account the quadrant
    theta_radians = math.atan2(math.degrees(force[1]), math.degrees(force[0]))
    # theta in [0, 180] for our dataset
    # 200 is a magic number, since the intensities of forces are now [200, 400]
    # print(f"force in: {force}")
    if theta_radians < 0:
        theta_radians += 2 * math.pi
    return theta_radians

def force_process2(force: np.ndarray) -> float:
    # Calculate the angle in radians using atan2, which takes into account the quadrant
    theta_radians = math.atan2(math.degrees(force[1]), math.degrees(force[0]))
    
    # Convert radians to degrees
    theta_degrees = math.degrees(theta_radians) / 720
    
    # Normalize the angle to be within [0, 360) degrees
    # if theta_degrees < 0:
    #     theta_degrees += 360
    
    # print(f"{math.degrees(force[1])}, {math.degrees(force[0])} -> {theta_degrees = }")
    return theta_degrees

def get_image_tensor(
    rgb_path: str, depth_path: str = None, seg_path: str = None,
    width = CAM_W, height = CAM_H
) -> Tuple[torch.TensorType, torch.TensorType, torch.TensorType]:
    rgb_image = Image.open(rgb_path).resize((width, height))
    norm_data = np.array(rgb_image.convert("RGBA"), dtype=np.float32) / 255.0
    # assume white_background = True
    arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + (1 - norm_data[:, :, 3:4])
    rgb = torch.tensor(arr, dtype=torch.float32).permute(2, 0, 1)
    dep = None if depth_path is None \
        else torch.tensor(np.array(Image.open(depth_path).resize((width, height)), dtype=np.float32) / 255.0)
    seg = None if seg_path is None \
        else torch.tensor(np.array(Image.open(seg_path).resize((width, height)), dtype=np.float32) / 255.0).permute(2, 0, 1)
    # seg=1 means background or floor, this means we change everything except the block to white-background
    # if seg is not None:
    #     rgb[seg == 1] = 1
    return rgb, dep, seg

def get_prev_images(path: str, prev: int = 4, width = CAM_W, height = CAM_H) -> torch.TensorType:
    if prev <= 0:
        return None
    infos = path.split('_')
    cur_step = infos[-1] # xxx.png
    cur_step = int(cur_step.split('.')[0])
    start_step = max(0, cur_step - prev)
    prev_imgs = torch.zeros((prev, 3, width, height))
    for i, _s in enumerate(range(start_step, cur_step)):
        new_path = "_".join(infos[:-1] + [f'{_s:03}.png'])
        assert os.path.exists(new_path)
        img = get_image_tensor(new_path, width=width, height=height)[0]
        # if prev > cur_step, skip the differences in output dimension
        _i = i + max(0, prev - cur_step)
        prev_imgs[_i] = img
    return prev_imgs


def get_next_images(path: str, next: int = 4, width = CAM_W, height = CAM_H) -> torch.TensorType:
    if next <= 0:
        return None
    infos = path.split('_')
    cur_step = infos[-1] # xxx.png
    cur_step = int(cur_step.split('.')[0])
    end_step = min(MAX_FRAME-1, cur_step + next)
    next_imgs = torch.zeros((next, 3, width, height))
    for i, _s in enumerate(range(cur_step, end_step)):
        # keep as 0
        if _s >= MAX_FRAME:
            continue
        new_path = "_".join(infos[:-1] + [f'{_s:03}.png'])
        assert os.path.exists(new_path), print(f"wrong, {_s = }")
        img = get_image_tensor(new_path, width=width, height=height)[0]
        next_imgs[i] = img
    return next_imgs
    

# use with cautious, lots of special case and magic numbers to maximize reading speed
def readCamShortParallel(
    path: str,
    mapper: Dict,
    force_idx: int,
    pos_idx: int,
    prev_frames: int = 0,
    next_frames: int = 0
) -> List[Camera]:
    cams: List[Camera] = []
    # print(f"{path = }, {force_idx = }, {pos_idx = }")
    with open(f"{path}/transforms_short.json") as json_file:
        contents: Dict = json.load(json_file)
    matrix = -np.linalg.inv(np.array(contents["transform_matrix"]))
    R = np.transpose(matrix[:3, :3])
    R[:, 0] = -R[:, 0]
    # force = force_process2(np.array(contents['force'])[3:5]) * 10  #  directly drop positions here
    force = np.array(contents['force'])[3:5] # / 100 # only keep xy rotations
    pose_idx = int(path.split('_')[2])
    # if pose_idx != 1:
    #     return []
    # print(f"{path = },   {path.split('_')[2]}")
    # breakpoint()
    force = force_process(force)
    def read_fn(idx_frame) -> Camera:
        frame_step, frame = idx_frame
        # print(f"{frame_step = }")
        if frame_step >= MAX_FRAME or frame_step < START_FRAME:
            return None
        rgb_path = f"{path}/{frame['file_path']}"
        # substitute r_ with d_ and s_ for depth and seg
        dep_path = re.sub(r'(r_)(\d+)(\.png)$', r'd_\2\3', rgb_path)
        seg_path = re.sub(r'(r_)(\d+)(\.png)$', r's_\2\3', rgb_path)
        dep_path = dep_path if os.path.isfile(dep_path) else None
        seg_path = seg_path if os.path.isfile(seg_path) else None
        rgb, dep, seg = get_image_tensor(rgb_path, dep_path, seg_path)
        prev_arr = get_prev_images(rgb_path, prev=prev_frames)
        next_arr = get_next_images(rgb_path, next=next_frames)
        # print(f"{frame_step = } | time = {mapper[frame['time']]}")
        cam = Camera(
            R=R,
            T=matrix[:3, 3],
            FoVx=contents['camera_angle_x'], # assume H and W are the same to save compute
            FoVy=contents['camera_angle_x'], # assume H and W are the same to save compute
            image=rgb,
            depth=dep,
            mask=seg,
            prev_frames=prev_arr,
            next_frames=next_arr,
            time=mapper[frame["time"]],
            frame_step=frame_step,
            force=force,
            full_force=[np.rint(np.degrees(np.array(contents['force'])[3])), np.rint(np.degrees(np.array(contents['force'])[4]))],
            force_idx=force_idx,
            pose_idx=pose_idx,
            pos_idx=pos_idx,
            unit_time=1/(MAX_FRAME - 1)
        )
        # print(f"{pos_idx = }")
        return cam

    with ThreadPool(10) as pool:
        cams = pool.map(read_fn, enumerate(contents["frames"]))
    return [cam for cam in cams if cam is not None]

def read_force_timeline(paths_train: List[str], paths_test: List[str]):
    # read from each force's subfolder
    time_lines = []
    for path in os.listdir(os.path.join(paths_train[0], 'train')):
        with open(os.path.join(paths_train[0], 'train', path, "transforms.json")) as json_file:
            train_json = json.load(json_file)
        time_lines += [frame["time"] for frame in train_json["frames"][START_FRAME:MAX_FRAME]]
        break  # just use 1 train cam and 1 test cam's time
    for path in os.listdir(os.path.join(paths_test[0], 'test')):
        with open(os.path.join(paths_test[0], 'test', path, "transforms.json")) as json_file:
            test_json = json.load(json_file)
        time_lines += [frame["time"] for frame in test_json["frames"][START_FRAME:MAX_FRAME]]
        break  # just use 1 train cam and 1 test cam's time
    time_lines = list(sorted(set(time_lines)))
    max_time_float = max(time_lines)
    timestamp_mapper = {t: t / max_time_float for t in time_lines}
    return timestamp_mapper, max_time_float

# def readForceSyntheticInfo(
#     paths_train: List[str],
#     paths_test: List[str],
#     n_train_cams: List[int],
#     n_test_cams: List[int],
#     white_background: bool
# ):
#     paths = paths_train + paths_test
#     timestamp_mapper, max_time = read_force_timeline(paths_train, paths_test)
#     def helper(path: str, i: int, split: str, force_idx: int) -> List[Camera]:
#         # force_setting is id of force
#         cam_path = os.path.join(path, f'{split}/cam_{i}')
#         # readCamerasFromTransforms, readCamerasFromShortTransforms
#         return readCamerasFromShortTransforms(cam_path, "transforms.json", timestamp_mapper, force_idx)

#     all_train_cams: List[Camera] = []
#     all_test_cams: List[Camera] = []
#     for force_idx, (_path, train_cams) in enumerate(zip(paths_train, n_train_cams)):
#         for i in tqdm(range(train_cams), desc='Reading Train'):
#             all_train_cams += helper(_path, i, 'train', force_idx)
#     for force_idx, (_path, test_cams) in enumerate(zip(paths_test, n_test_cams)):
#         for i in tqdm(range(test_cams), desc='Reading Test'):
#             all_test_cams += helper(_path, i, 'test', force_idx)

#     nerf_normalization = getNerfppNorm(all_train_cams)
#     ply_path = os.path.join(paths[0], "fused.ply") # NOTE: PLACEHOLDER?
#     num_pts = 2000
#     # We create random points inside the bounds of the synthetic Blender scenes
#     xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
#     shs = np.random.random((num_pts, 3)) / 255.0
#     pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
#     scene_info = SceneInfo(
#         point_cloud=pcd,
#         train_cameras=all_train_cams,
#         test_cameras=all_test_cams,
#         nerf_normalization=nerf_normalization,
#         ply_path=ply_path,
#         maxtime=max_time
#     )
#     return scene_info

def readForceSyntheticInfo2(
    paths_train: List[str],
    paths_test: List[str],
    n_train_cams: List[int],
    n_test_cams: List[int],
    prev_frames: int = 0,
    next_frames: int = 0
):
    paths = paths_train + paths_test
    timestamp_mapper, maxtime = read_force_timeline(paths_train, paths_test)

    def helper_train(args) -> List[Camera]:
        path, i, force_idx = args
        cam_path = os.path.join(f"{path}/train", os.listdir(f"{path}/train")[i])
        return readCamShortParallel(cam_path, timestamp_mapper, force_idx, i, prev_frames, next_frames)
    
    def helper_test(args) -> List[Camera]:
        path, i, force_idx = args
        cam_path = os.path.join(f"{path}/test", os.listdir(f"{path}/test")[i])
        return readCamShortParallel(cam_path, timestamp_mapper, force_idx, i, prev_frames, next_frames)

    time_start = time.time()
    # extend the paths for parallelization
    # i.e. [path1, path2], [10, 15] 
    #    -> [[path1] * 10, [path2]*15]
    #    -> [...path1..., ...path2...]
    paths_train_extended = []
    paths_test_extended = []
    force_idx_train_extended = []
    force_idx_test_extended = []
    cam_idx_train_extended = []
    cam_idx_test_extended = []
    for i, n in enumerate(n_train_cams):
        paths_train_extended.extend([paths_train[i]] * n)
        force_idx_train_extended.extend([i] * n)
        cam_idx_train_extended.extend(list(range(n)))
    for i, n in enumerate(n_test_cams):
        paths_test_extended.extend([paths_test[i]] * n)
        force_idx_test_extended.extend([i] * n)
        cam_idx_test_extended.extend(list(range(n)))

    with ThreadPool(10) as pool:
        train_cams: List[List[Camera]] = pool.map(
            helper_train,
            zip(paths_train_extended, cam_idx_train_extended, force_idx_train_extended)
        )
    # get flatten from nested parallel
    train_cams: List[Camera] = [c for cams in train_cams for c in cams if c is not None]
    # train_cams = list(sorted(train_cams, key = lambda c: (c.force_idx, c.pos_idx, c.time)))
    
    with ThreadPool(10) as pool:
        test_cams: List[List[Camera]] = pool.map(
            helper_test,
            zip(paths_test_extended, cam_idx_test_extended, force_idx_test_extended)
        )
    test_cams: List[Camera] = [c for cams in test_cams for c in cams if c is not None]
    # test_cams = list(sorted(test_cams, key = lambda c: (c.force_idx, c.pos_idx, c.time)))
    time_end = time.time()
    # for c in test_cams: print(c.force_idx, c.pos_idx, c.time)
    print(f"parallelized total time: {time_end - time_start}")
    # breakpoint()
    num_pts = 2000
    # We create random points inside the bounds of the synthetic Blender scenes
    xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
    shs = np.random.random((num_pts, 3)) / 255.0
    scene_info = SceneInfo(
        point_cloud=BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3))),
        train_cameras=train_cams,
        test_cameras=test_cams,
        nerf_normalization=getNerfppNorm(train_cams),
        ply_path=os.path.join(paths[0], "fused.ply"), # NOTE: PLACEHOLDER?
        maxtime=maxtime
    )
    return scene_info

# def format_infos(dataset,split):
#     # loading
#     cameras = []
#     image = dataset[0][0]
#     if split == "train":
#         for idx in tqdm(range(len(dataset))):
#             image_path = None
#             image_name = f"{idx}"
#             time = dataset.image_times[idx]
#             # matrix = np.linalg.inv(np.array(pose))
#             R,T = dataset.load_pose(idx)
#             FovX = focal2fov(dataset.focal[0], image.shape[1])
#             FovY = focal2fov(dataset.focal[0], image.shape[2])
#             cameras.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
#                                 image_path=image_path, image_name=image_name, width=image.shape[2], height=image.shape[1],
#                                 time = time, mask=None))

#     return cameras

# def format_render_poses(poses,data_infos):
#     cameras = []
#     tensor_to_pil = transforms.ToPILImage()
#     len_poses = len(poses)
#     times = [i/len_poses for i in range(len_poses)]
#     image = data_infos[0][0]
#     for idx, p in tqdm(enumerate(poses)):
#         # image = None
#         image_path = None
#         image_name = f"{idx}"
#         time = times[idx]
#         pose = np.eye(4)
#         pose[:3,:] = p[:3,:]
#         # matrix = np.linalg.inv(np.array(pose))
#         R = pose[:3,:3]
#         R = - R
#         R[:,0] = -R[:,0]
#         T = -pose[:3,3].dot(R)
#         FovX = focal2fov(data_infos.focal[0], image.shape[2])
#         FovY = focal2fov(data_infos.focal[0], image.shape[1])
#         cameras.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
#                             image_path=image_path, image_name=image_name, width=image.shape[2], height=image.shape[1],
#                             time = time, mask=None))
#     return cameras

def add_points(pointsclouds, xyz_min, xyz_max):
    add_points = (np.random.random((100000, 3)))* (xyz_max-xyz_min) + xyz_min
    add_points = add_points.astype(np.float32)
    addcolors = np.random.random((100000, 3)).astype(np.float32)
    addnormals = np.random.random((100000, 3)).astype(np.float32)
    new_points = np.vstack([pointsclouds.points,add_points])
    new_colors = np.vstack([pointsclouds.colors,addcolors])
    new_normals = np.vstack([pointsclouds.normals,addnormals])
    pointsclouds=pointsclouds._replace(points=new_points)
    pointsclouds=pointsclouds._replace(colors=new_colors)
    pointsclouds=pointsclouds._replace(normals=new_normals)
    return pointsclouds

# def setup_camera(w, h, k, w2c, near=0.01, far=100):
#     from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera
#     fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
#     w2c = torch.tensor(w2c).cuda().to(dtype=torch.float32)
#     cam_center = torch.inverse(w2c)[:3, 3]
#     w2c = w2c.unsqueeze(0).transpose(1, 2)
#     opengl_proj = torch.tensor([[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
#                                 [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
#                                 [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
#                                 [0.0, 0.0, 1.0, 0.0]]).cuda().to(dtype=torch.float32).unsqueeze(0).transpose(1, 2)
#     full_proj = w2c.bmm(opengl_proj)
#     cam = Camera(
#         image_height=h,
#         image_width=w,
#         tanfovx=w / (2 * fx),
#         tanfovy=h / (2 * fy),
#         bg=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
#         scale_modifier=1.0,
#         viewmatrix=w2c,
#         projmatrix=full_proj,
#         sh_degree=0,
#         campos=cam_center,
#         prefiltered=False,
#         debug=True
#     )
#     return cam

# def plot_camera_orientations(cam_list, xyz):
#     import matplotlib.pyplot as plt
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     # ax2 = fig.add_subplot(122, projection='3d')
#     # xyz = xyz[xyz[:,0]<1]
#     threshold=2
#     xyz = xyz[(xyz[:, 0] >= -threshold) & (xyz[:, 0] <= threshold) &
#                          (xyz[:, 1] >= -threshold) & (xyz[:, 1] <= threshold) &
#                          (xyz[:, 2] >= -threshold) & (xyz[:, 2] <= threshold)]

#     ax.scatter(xyz[:,0],xyz[:,1],xyz[:,2],c='r',s=0.1)
#     for cam in tqdm(cam_list):
#         # 提取 R 和 T
#         R = cam.R
#         T = cam.T

#         direction = R @ np.array([0, 0, 1])

#         ax.quiver(T[0], T[1], T[2], direction[0], direction[1], direction[2], length=1)

#     ax.set_xlabel('X Axis')
#     ax.set_ylabel('Y Axis')
#     ax.set_zlabel('Z Axis')
#     plt.savefig("output.png")

