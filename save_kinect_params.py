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


DEPTH_MAX_VAL = 4500.0
ASSET_DIR = "assets"

if len(sys.argv) < 2:
    print("Too few arguments")
    exit(-1)

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
ir_params = device.ir_camera_params
ir_params = {"fx": ir_params.fx, "fy": ir_params.fy, "cx": ir_params.cx, "cy": ir_params.cy, "k1": ir_params.k1, "k2": ir_params.k2, "k3": ir_params.k3, "p1": ir_params.p1, "p2": ir_params.p2}
color_params = device.color_camera_params
color_params = {"fx": color_params.fx, "fy": color_params.fy, "cx": color_params.cx, "cy": color_params.cy}
kinect_params = {"IR":ir_params, "COLOR":color_params}
f = open(path.join(ASSET_DIR, sys.argv[1]), "w")
json.dump(kinect_params, f)
f.close()
