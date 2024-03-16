# import pycolmap

# database_path = "/home/baiting/4DGaussians/data_new/scene2_dir_test6/colmap/database.db"
# image_dir = "/home/baiting/4DGaussians/data_new/scene2_dir_test6/colmap/images"
# pycolmap.extract_features(database_path, image_dir, sift_options={"max_num_features": 512})
# # equivalent to
# # ops = pycolmap.SiftExtractionOptions()
# # ops.max_num_features = 512
# # pycolmap.extract_features(database_path, image_dir, sift_options=ops)



import open3d as o3d
import numpy as np
from PIL import Image
import json

example_path = "/home/baiting/4DGaussians/data_new/scene2_dir_test6/test/cam_0/r_000.png"
example_json = "/home/baiting/4DGaussians/data_new/scene2_dir_test6/test/cam_0/transforms_short.json"
example_intrinsic = np.array([
    [833.3335240681967, 0.0, 299.5],
    [0.0, 833.3335240681967, 299.5],
    [0.0, 0.0, 1.0]
])

intrinsic = o3d.cuda.pybind.camera.PinholeCameraIntrinsic(
    600,
    600,
    example_intrinsic
)

full = np.array(Image.open(example_path))
color = full[:, :, :3]
depth = full[:, :, 3:]
depth_as_img = o3d.geometry.Image((depth).astype(np.uint8))
color_as_img = o3d.geometry.Image((color).astype(np.uint8))

rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color_as_img, depth_as_img, convert_rgb_to_intensity = False)
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

# flip the orientation, so it looks upright, not upside-down
pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])


print(pcd)
print(type(pcd))
print("123")
o3d.visualization.draw_geometries([pcd])
print("456")
# o3d.visualizadraw_geometries([pcd])    # visualize the point cloud