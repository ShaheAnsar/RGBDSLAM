import numpy as np
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
