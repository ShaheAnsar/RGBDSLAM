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
depth_undistorted, rgb_registered = device.registration.apply(rgb_frame, depth_frame, enable_filter=True)
depth_undistorted = depth_undistorted.to_array()
print(np.max(depth_undistorted))
rgb_registered = rgb_registered.to_array()
cv2.imwrite(path.join(ASSET_DIR, sys.argv[1] + "_rgb.png"), rgb_registered)
with open(path.join(ASSET_DIR, sys.argv[1] + "_depth.npy"), "wb") as f:
    np.save(f, depth_undistorted)
#cv2.imwrite(path.join(ASSET_DIR, sys.argv[1] + "_depth.png"), depth_undistorted)
#point_cloud, rgb_cloud = convert_rgb_depth_to_pcl(device, rgb_frame, depth_frame)
#pts = vedo.Points(point_cloud, c=rgb_cloud)
#vedo.show(pts, __doc__, axes=True)
