from freenect2 import Device, FrameType
import numpy as np
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import proj3d
from kinect_preprocess import *
from mobrgbd import MobRGBD
import os
from os import path
import sys
import json
import open3d as o3d
from frame_sampler import FrameSampler
from pprint import pprint
from time import perf_counter, perf_counter_ns, process_time
from frame_listener import FrameListener
from pose_graph import Node, Edge, PoseGraph



DEPTH_MAX_VAL = 4500.0
ASSET_DIR = "assets"
PARAMS_NAME = "params.json"
f = open(path.join(ASSET_DIR, PARAMS_NAME))
ir_params = json.load(f)
f.close()
ir_params = ir_params["IR"]
frame_callback = FrameListener()
device = Device()
fr = FrameSampler(device)
t = process_time()
pg = PoseGraph()
curr_pose = np.identity(4)
loop_closure_rmse_thresh = 0.05
loop_clousre_fitness_thresh = 0.1
odometry_fitness_thresh = 0.02
with device.running(frame_callback):
    frame_set = [None, None]
    i = 0
    for type_, frame in frame_callback:
        if type_ is None:
            continue
        if type_ is FrameType.Depth:
            frame_set[1] = frame
        if type_ is FrameType.Color:
            frame_set[0] = frame
        if None not in frame_set:
            dt = process_time() - t
            t = process_time()
            print(f"Time for frame: {dt}")
            r = None
            #print(r)
            # For the first iteration just insert the first node, and assume position is 0
            if i == 0:
                pcd1, _ = fr.preprocess_frame(frame_set)
                fr.push_frame(frame_set)
                n = Node(0, pcd1, curr_pose)
                pg.insert_node(n)
            # Only use the recent frames if the uniform set isn't ready for processing yet
            else:
                # Visual Odometry
                r = fr.compare_and_push_frame(frame_set)
                pcl_id = r[2][0]
                transform = r[2][2].transformation
                rmse = r[2][2].inlier_rmse
                fitness = r[2][2].fitness
                n_prev_i, n_prev = pg.get_node_and_index(pcl_id)
                curr_pose = np.dot(transform, n_prev.pose)
                pprint(curr_pose)
                n = Node(r[1], r[0], curr_pose)
                pg.insert_node(n)
                e = None
                if fitness >= odometry_fitness_thresh:
                    e = Edge(n_prev_i, len(pg.nodes) - 1, transform, (rmse, fitness))
                else:
                    e = Edge(n_prev_i, len(pg.nodes) - 1, transform, (rmse, fitness), Edge.ETYPE_BROKEN)
                pg.add_edge(e)

            if r is not None and i >= 12 and r[3][2] is not None and r[3][2].fitness >= loop_clousre_fitness_thresh and r[3][2].inlier_rmse <= loop_closure_rmse_thresh:
                # Loop closure
                pcl_id = r[3][0]
                transform = r[3][2].transformation
                rmse = r[3][2].inlier_rmse
                fitness = r[3][2].fitness
                n_prev_i, n_prev = pg.get_node_and_index(pcl_id)
                e = Edge(n_prev_i, len(pg.nodes) - 1, transform, (rmse, fitness), Edge.ETYPE_LOOP)
                pg.add_edge(e)

            if i >= 100:
                break

            frame_set = [None, None]
            i += 1

pg.store("assets/graph.pickle")
pg = PoseGraph.load("assets/graph.pickle")
pg.construct_factor_graph()
r = pg.optimize()
print(pg.nodes)
pg.visualize()
cv2.waitKey(0)
#pg.visualize_edges()
pg.optimized_visualize()
cv2.waitKey(0)
print(r)
##depth_undistorted, rgb_registered = device.registration.apply(rgb_frame, depth_frame, enable_filter=True)
##print(depth_undistorted.bytes_per_pixel, depth_undistorted.width, depth_undistorted.height)
##depth_arr = depth_frame.to_array()
##point_cloud = []
##for j in range(depth_arr.shape[0]):
##    for i in range(depth_arr.shape[1]):
##        point_cloud.append(get_xyz(depth_arr, j, i, ir_params))
#point_cloud, rgb_cloud = convert_rgb_depth_to_pcl(device, rgb_frame, depth_frame)
##pts = vedo.Points(point_cloud, c=rgb_cloud)
##pts = vedo.Points(point_cloud, c=(1.0, 0.0, 0.0))
##vedo.show(pts, __doc__, axes=True)
#pcd = o3d.geometry.PointCloud()
#pcd.points = o3d.utility.Vector3dVector(point_cloud)
#rgb_cloud = rgb_cloud.astype(np.float64)
#pcd.colors = o3d.utility.Vector3dVector(rgb_cloud/255.0)
#o3d.visualization.draw_geometries([pcd])


