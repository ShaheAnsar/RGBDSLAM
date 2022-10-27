from freenect2 import Device, FrameType
import numpy as np
import cv2
import vedo
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import proj3d
from kinect_preprocess import *


DEPTH_MAX_VAL = 4500.0

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
#except as e:
#    print("Issue encountered")
#    print(e)
#    exit(-1)

#cv2.destroyAllWindows()
depth_undistorted, rgb_registered = device.registration.apply(rgb_frame, depth_frame, enable_filter=False)
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
print(point_cloud.shape)
print(point_cloud)
pts = vedo.Points(point_cloud, c=rgb_cloud)
vedo.show(pts, __doc__, axes=True)
