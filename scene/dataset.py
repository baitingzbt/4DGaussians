from torch.utils.data import Dataset
from scene.cameras import Camera
from scene.dataset_readers import CameraInfo
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal, focal2fov
import torch
from utils.camera_utils import loadCam
from utils.graphics_utils import focal2fov
from typing import List

class FourDGSdataset(Dataset):

    def __init__(
        self,
        dataset,
        args,
        dataset_type
    ):
        self.dataset = dataset
        self.args = args
        self.dataset_type=dataset_type

    def __getitem__(self, index):
        if self.dataset_type == 'PanopticSports':
            return self.dataset[index]

        if self.dataset_type in ['blender']:
            caminfo: CameraInfo = self.dataset[index]
            image = caminfo.image
            R, T = caminfo.R, caminfo.T
            FovX, FovY = caminfo.FovX, caminfo.FovY
            time = caminfo.time
            mask = caminfo.mask
            force = caminfo.force
        else:
            image, w2c, time = self.dataset[index]
            R,T = w2c
            FovX = focal2fov(self.dataset.focal[0], image.shape[2])
            FovY = focal2fov(self.dataset.focal[0], image.shape[1])
            mask=None
        return Camera(
            colmap_id=index, R=R, T=T, FoVx=FovX, FoVy=FovY,
            image=image, gt_alpha_mask=None, image_name=f"{index}",
            uid=index, data_device=torch.device("cuda"), time=time, mask=mask, force=force
        )

    def __len__(self):
        return len(self.dataset)
