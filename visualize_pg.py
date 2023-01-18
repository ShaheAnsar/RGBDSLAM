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
pg.visualize()
for i in pg.edges:
    print(i.decompose())
cv2.waitKey(0)
