from freenect2 import Device, FrameType
import numpy as np
import cv2
#import vedo
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
pg = PoseGraph.load("assets/graph.pickle")
print("Unoptimized Map")
pg.visualize(voxel_size=0.01)
cv2.waitKey(0)
if "-e" in sys.argv:
    print("Edges")
    if "-c" in sys.argv:
        pg.visualize_edges(uniform_color=False)
    else:
        pg.visualize_edges()
print("Optimized map")
pg.construct_factor_graph()
pg.optimize()
pcd = pg.optimized_visualize(voxel_size=0.01, uniform_color=False)
cv2.waitKey(0)
grid_map = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, 0.05)
o3d.visualization.draw_geometries([grid_map])
