import numpy as np
import vedo as v
import pyunpack as pu
import os

class MobRGBD:
    def __init__(self, depth_path):
        self.depth_path = depth_path
        self.w = 512
        self.h = 424

    @staticmethod
    def __mobrgbd__find_depth_raw(path):
        for dirpath, _, filenames in os.walk(path):
            if "depth.raw" in filenames:
                return os.path.join(dirpath, "depth.raw")

    def init(self):
        depth_path = self.depth_path
        ar = pu.Archive(os.path.join(depth_path, "depth.raw.7z"))
        output_dir = "extracted"
        output_dir = os.path.join(self.depth_path, output_dir)
        try:
            os.mkdir(output_dir)
        except FileExistsError:
            pass
            
        ar.extractall(os.path.join(depth_path, output_dir))
        self.depthfile = self.__mobrgbd__find_depth_raw(os.path.join(depth_path,
                                                                output_dir))
        self.f = open(self.depthfile, "rb")

    def get_frame(self):
        h = self.h
        w = self.w
        f = self.f
        bslice = bytearray(f.read(h*w*2))
        frame = np.frombuffer(bslice, dtype=np.uint16)
        frame = frame.reshape(h, w)
        return frame
