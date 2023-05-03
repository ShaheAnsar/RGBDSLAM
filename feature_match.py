# This script matches features and provides a transform between two point clouds


from freenect2 import Device, FrameType
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
from os import path
import math as m
import random as r
from time import perf_counter
import kinect_preprocess as p
import internal_math as im
import json
import open3d as o3d
import copy


ASSET_DIR = "assets"
FR1_NAME = "fr1"
FR2_NAME = "fr2"
PARAMS_NAME = "params.json"
FR1_PATH = path.join(ASSET_DIR, FR1_NAME)
FR2_PATH = path.join(ASSET_DIR, FR2_NAME)
#dev = Device()
#dev.start()
#dev.close()

# Import the data
fr1_rgb = cv2.imread(FR1_PATH + "_rgb.png")
fr2_rgb = cv2.imread(FR2_PATH + "_rgb.png")
fr1_depth = np.load(FR1_PATH + "_depth.npy")
fr2_depth = np.load(FR2_PATH + "_depth.npy")
f = open(path.join(ASSET_DIR, PARAMS_NAME))
params = json.load(f)
f.close()
ir_params = params["IR"]
rgb_params = params["COLOR"]


o3d_rgb1 = o3d.geometry.Image(cv2.cvtColor(fr1_rgb, cv2.COLOR_BGR2RGB))
o3d_rgb2 = o3d.geometry.Image(cv2.cvtColor(fr2_rgb, cv2.COLOR_BGR2RGB))
o3d_depth1 = o3d.geometry.Image(fr1_depth)
o3d_depth2 = o3d.geometry.Image(fr2_depth)

rgbd1 = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_rgb1, o3d_depth1, convert_rgb_to_intensity=False, depth_trunc=5.0)
rgbd2 = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_rgb2, o3d_depth2, convert_rgb_to_intensity=False, depth_trunc=5.0)
pcd1 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd1, o3d.camera.PinholeCameraIntrinsic(512,424,ir_params["fx"], ir_params["fy"], ir_params["cx"], ir_params["cy"]))
pcd2 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd2, o3d.camera.PinholeCameraIntrinsic(512,424,ir_params["fx"], ir_params["fy"], ir_params["cx"], ir_params["cy"]))

fr1_rgb = cv2.cvtColor(fr1_rgb, cv2.COLOR_BGR2RGB)

o3d.visualization.draw_geometries([pcd1])
o3d.visualization.draw_geometries([pcd2])

def preprocess_pcl(pcl, voxel_size):
    down = pcl.voxel_down_sample(voxel_size)
    down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2,max_nn=30))
    down_features = o3d.pipelines.registration.compute_fpfh_feature(down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*5, max_nn=100))
    return down, down_features

def prepare_data(pcd_source, pcd_target, voxel_size):
    source_down, source_features = preprocess_pcl(pcd_source, voxel_size)
    target_down, target_features = preprocess_pcl(pcd_target, voxel_size)
    return source_down, source_features, target_down, target_features

def draw_registration(pcd1, pcd2, tr, uniform_colors=True):
    pcd1copy = copy.deepcopy(pcd1)
    pcd2copy = copy.deepcopy(pcd2)
    if uniform_colors:
        pcd1copy.paint_uniform_color([1.0, 0.0, 0.0])
        pcd2copy.paint_uniform_color([0.0, 1.0, 0.0])
    pcd1copy.transform(tr)
    o3d.visualization.draw_geometries([pcd1copy, pcd2copy])
# Global Registration
t = perf_counter()
voxel_size = 0.05
distance_thresh = 1.5*voxel_size
pcd1down, pcd1feat, pcd2down, pcd2feat = prepare_data(pcd1, pcd2, voxel_size)
# RANSAC Registraion

result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(pcd1down, pcd2down, pcd1feat, pcd2feat, True, distance_thresh,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_thresh)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))

# FAST Registration, is actually slow don't use
#result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
#        pcd1down, pcd2down, pcd1feat, pcd2feat, o3d.pipelines.registration.FastGlobalRegistrationOption(
#            maximum_correspondence_distance=distance_thresh))
t = perf_counter() - t
print(result)
print(result.transformation)
print(t)
cv_tr = np.array([[0.968581, 0.035594, -0.24614, 0.0769454],
                  [-0.0344965, 0.999366, 0.00877052, -0.0768757],
                  [0.246296, -3.99053e-6, 0.969195, 0.00132942],
                  [0, 0, 0, 1]])
draw_registration(pcd1, pcd2, np.identity(4))
draw_registration(pcd1, pcd2, cv_tr, uniform_colors=False)
draw_registration(pcd1, pcd2, result.transformation, uniform_colors=False)
