import numpy as np
import vedo as v
import pyunpack as pu
import os
import shutil
import cv2

class MobRGBD:
    def __init__(self, depth_path):
        self.depth_path = depth_path
        self.w = 512
        self.h = 424
        self.DEPTH_MAX_VAL = 4500


    @staticmethod
    def __mobrgbd__find_depth_raw(path):
        for dirpath, _, filenames in os.walk(path):
            if "depth.raw" in filenames:
                return os.path.join(dirpath, "depth.raw")

    def init(self):
        depth_path = self.depth_path
        ar = pu.Archive(os.path.join(depth_path, "depth.raw.7z"))
        self.output_dir = "extracted"
        self.output_dir = os.path.join(self.depth_path, self.output_dir)
        try:
            os.mkdir(self.output_dir)
        except FileExistsError:
            pass
            
        ar.extractall(self.output_dir)
        self.depthfile = self.__mobrgbd__find_depth_raw(self.output_dir)
        self.f = open(self.depthfile, "rb")

    def cleanup(self):
        shutil.rmtree(self.output_dir)

    def get_frame(self):
        h = self.h
        w = self.w
        f = self.f
        bslice = bytearray(f.read(h*w*2))
        frame = np.frombuffer(bslice, dtype=np.uint16)
        frame = frame.reshape(h, w)
        return frame

    def play_all_frames(self):
        while True:
            frame = self.get_frame() * 10
            cv2.imshow("frame", frame)
            cv2.waitKey(10)
