import freenect2
import numpy as np
import sys



class FrameListener:
    def __init__(self, len=3):
        self.l = []
        self.len=len

    def __call__(self, frame, frame_type):
        if len(self.l) < self.len:
            self.l.append((frame, frame_type))
        else:
            # Keep on updating the last frame if the list is filled
            self.l[-1] = (frame, frame_type)

    def get(self, timeout=False):
        # Pop the first element in the list
        if len(self.l) > 0:
            ret = self.l.pop(0)
            return ret
        else:
            return None, None

    def __iter__(self):
        def iterator():
            while True:
                yield self.get()
        return iterator()
