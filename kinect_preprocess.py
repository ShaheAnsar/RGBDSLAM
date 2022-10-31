import numpy as np
import math as m
import cv2
from freenect2 import Device, FrameType


def convert_rgb_depth_to_pcl(device, rgb_frame, depth_frame):
    depth_undistorted, rgb_registered = device.registration.apply(rgb_frame, depth_frame, enable_filter=True)
    point_cloud = device.registration.get_points_xyz_array(depth_undistorted)
    indices = ~np.isnan(point_cloud)
    point_cloud = point_cloud[indices]
    rgb_cloud = rgb_registered.to_array()
    cv2.imshow("RGB Reg", rgb_cloud)
    cv2.waitKey(0)
    rgb_cloud = rgb_cloud[:,:,:3]
    rgb_cloud = cv2.cvtColor(rgb_cloud, cv2.COLOR_BGR2RGB)
    rgb_cloud = rgb_cloud[indices]
    s = point_cloud.shape
    point_cloud = point_cloud.reshape((s[0]//3, 3))
    s = rgb_cloud.shape
    rgb_cloud = rgb_cloud.reshape((s[0]//3, 3))
    return (point_cloud, rgb_cloud)

def get_depth(depth_undistorted, x, y):
    x_l = m.floor(x)
    y_l = m.floor(y)
    x_u = m.ceil(x)
    y_u = m.ceil(y)
    # ul - upper left, lr - lower right
    # ll - lower left, ur - upper right
    depth_ul = depth_undistorted[y_l, x_l]
    depth_lr = depth_undistorted[y_u, x_u]
    depth_ur = depth_undistorted[y_l, x_u]
    depth_ll = depth_undistorted[y_u, x_l]
    # Interpolation const in the y dir
    t_y = y - y_l
    # Interpolation const in the x dir
    t_x = x - x_l
    depth_y = t_y*depth_ll + (1 - t_y)*depth_ul
    depth_x = t_x*depth_lr + (1-t_x)*depth_ll
    return 0.5*depth_x + 0.5*depth_y


