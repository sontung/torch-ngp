import pickle
import open3d as o3d
import gc
import numpy as np
from nerf.utils import make_point_cloud_from_nerf, return_cam_mesh_with_pose


with open('debug.pickle', 'rb') as handle:
    data = pickle.load(handle)
[all_ro, all_rd, all_depths, all_names, all_sizes, all_xyz, all_pose, all_rgb] = data
all_xyz = np.stack(all_xyz, axis=0)
all_rgb = np.stack(all_rgb, axis=0).reshape((all_xyz.shape[0], -1, 3))
del all_ro, all_rd, all_depths, all_names, all_sizes
gc.collect()

all_xyz = all_xyz[0, :]
all_rgb = all_rgb[0, :]

all_rgb = np.clip(all_rgb, 0, 1)
point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(all_xyz.reshape((-1, 3))))
point_cloud.colors = o3d.utility.Vector3dVector(all_rgb.reshape((-1, 3)))

vis = o3d.visualization.Visualizer()
vis.create_window(width=1920, height=1025)

vis.add_geometry(point_cloud)
cameras = return_cam_mesh_with_pose(all_pose)
for c in cameras:
    vis.add_geometry(c)

vis.run()
vis.destroy_window()
