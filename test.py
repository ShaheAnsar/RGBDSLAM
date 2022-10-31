from freenect2 import Device, FrameType
import numpy as np
import cv2
import vedo
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import proj3d
from kinect_preprocess import *
from mobrgbd import MobRGBD
import os
from os import path
import sys
import json
import open3d as o3d



DEPTH_MAX_VAL = 4500.0
ASSET_DIR = "assets"
PARAMS_NAME = "params.json"
f = open(path.join(ASSET_DIR, PARAMS_NAME))
ir_params = json.load(f)
f.close()
ir_params = ir_params["IR"]
device = Device()
depth_frame = None
rgb_frame = None
with device.running():
    for type_, frame in device:
        if type_ is FrameType.Depth:
            cv2.imshow("Depth Stream", frame.to_array()/DEPTH_MAX_VAL)
            depth_frame = frame
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        if type_ is FrameType.Color:
            cv2.imshow("Color Image", frame.to_array())
            rgb_frame = frame
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
#depth_undistorted, rgb_registered = device.registration.apply(rgb_frame, depth_frame, enable_filter=True)
#print(depth_undistorted.bytes_per_pixel, depth_undistorted.width, depth_undistorted.height)
#depth_arr = depth_frame.to_array()
#point_cloud = []
#for j in range(depth_arr.shape[0]):
#    for i in range(depth_arr.shape[1]):
#        point_cloud.append(get_xyz(depth_arr, j, i, ir_params))
point_cloud, rgb_cloud = convert_rgb_depth_to_pcl(device, rgb_frame, depth_frame)
#pts = vedo.Points(point_cloud, c=rgb_cloud)
#pts = vedo.Points(point_cloud, c=(1.0, 0.0, 0.0))
#vedo.show(pts, __doc__, axes=True)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud)
rgb_cloud = rgb_cloud.astype(np.float64)
pcd.colors = o3d.utility.Vector3dVector(rgb_cloud/255.0)
o3d.visualization.draw_geometries([pcd])
o3d.visualization.draw_geometries([pcd])
