# from torch.utils.data import Dataset
# from scene.cameras import Camera
# from scene.dataset_readers import CameraInfo
# import torch

# class FourDGSdataset(Dataset):

#     def __init__(
#         self,
#         dataset,
#         args,
#         dataset_type
#     ):
#         self.dataset = dataset
#         self.args = args
#         self.dataset_type=dataset_type

#     def __getitem__(self, index):

#         caminfo: CameraInfo = self.dataset[index]
#         # if time != 0:
#         #     prev_caminfo: CameraInfo = self.dataset[index-1]
#         #     prev_image = prev_caminfo.image
#         #     prev_time = prev_caminfo.time
#         # else:
#         #     prev_image = torch.zeros_like(image)
#         #     prev_time = -1
#         # print(f"-----> time = {time}, prev_time = {prev_time}, diff = {prev_time - time}")
#         return Camera(
#             colmap_id=index, R=caminfo.R, T=caminfo.T, FoVx=caminfo.FovX, FoVy=caminfo.FovY, # prev_image=prev_image,
#             image=caminfo.image, gt_alpha_mask=None, image_name=f"{index}",
#             uid=index, data_device=torch.device("cuda"),
#             time=caminfo.time, mask=caminfo.mask, force=caminfo.force, force_idx=caminfo.force_idx
#         )

#     def __len__(self):
#         return len(self.dataset)
