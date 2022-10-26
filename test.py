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
        frame = frame.to_array()
        if type_ is FrameType.Depth:
            cv2.imshow("Depth Stream", frame/DEPTH_MAX_VAL)
            depth_frame = frame
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        #if type_ is FrameType.Color:
        #    cv2.imshow("Color Image", frame/255.0)
        #    rgb_frame = frame
#except as e:
#    print("Issue encountered")
#    print(e)
#    exit(-1)

cv2.destroyAllWindows()
depth_undistorted, rgb_registered = device.registration.apply(rgb_frame, depth_frame)
point_cloud = device.registration.get_points_xyz_array(depth_undistorted)
point_cloud.reshape((point_cloud.shape[0]*point_cloud.shape[1], 3))
vedo.show(pts, __doc__, axes=True)
#shape = frame.shape
#x = []
#y = []
#z = []
#for v in range(shape[0]):
#    for u in range(shape[1]):
#        x_, y_, z_ = convert_from_uvd(u, v, frame[v][u])
#        x.append(x_)
#        y.append(y_)
#        z.append(z_)
#x = np.array(x)
#y = np.array(y)
#z = np.array(z)
#print(np.max(x))
#print(np.max(y))
#print(np.max(z))
#point_cloud = np.c_[x, y, z]
#print(point_cloud.shape)
#print(point_cloud)
#pts = vedo.Points(point_cloud, c=(0, 0.5, 0.3))
#vedo.show(pts, __doc__, axes=True)
#print("Ended")
#fig = plt.figure(figsize=(8,8))
#ax = fig.add_subplot(111, projection="3d")
#ax.scatter(x, y, z)
#plt.show()
