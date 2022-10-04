from freenect2 import Device, FrameType
import numpy as np
import cv2
import vedo
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import proj3d
from kinect_preprocess import *


DEPTH_MAX_VAL = 4500.0

device = Device()
try:
    with device.running():
        for type_, frame in device:
            frame = frame.to_array()
            if type_ is FrameType.Depth:
                cv2.imshow("Depth Stream", frame/DEPTH_MAX_VAL)
#                cv2.imsave("Depth Image", frame/DEPTH_MAX_VAL)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
#           # if type_ is FrameType.Color:
#           #     cv2.imsave("Color Image", frame/255)
except:
    print("Issue encountered")
    exit(-1)

cv2.destroyAllWindows()
shape = frame.shape
x = []
y = []
z = []
for v in range(shape[0]):
    for u in range(shape[1]):
        x_, y_, z_ = convert_from_uvd(u, v, frame[v][u])
        x.append(x_)
        y.append(y_)
        z.append(z_)
x = np.array(x)
y = np.array(y)
z = np.array(z)
print(np.max(x))
print(np.max(y))
print(np.max(z))
point_cloud = np.c_[x, y, z]
print(point_cloud.shape)
print(point_cloud)
pts = vedo.Points(point_cloud, c=(0, 0.5, 0.3))
vedo.show(pts, __doc__, axes=True)
print("Ended")
#fig = plt.figure(figsize=(8,8))
#ax = fig.add_subplot(111, projection="3d")
#ax.scatter(x, y, z)
#plt.show()
