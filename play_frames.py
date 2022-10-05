from mobrgbd import *
import numpy as np
import cv2
import sys


m = MobRGBD("/home/shahe/hdd/minor-proj/depth/MobileRGBD/Corridor/Traj_132_-30_Corridor_0.3/depth/")
m.init()
m.play_all_frames()
m.destroy()
