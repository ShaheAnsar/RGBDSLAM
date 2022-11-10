import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import cv2
import time as tm


#Maintains 15 frames, of which 12 are uniformly sampled and the remainig 3 are the most recent ones
class FrameSampler:
    def __init__(self, uni_len=12, rec_len=3):
        self.frames = []
        self.rframes = []
        self.next_frame_id = 0
        self.ulen = uni_len
        self.rlen = rec_len
        # Used to denote the index of the last frame appended
        self.curr_i = 0
        # Time diff used to sample frames
        self.time_thresh = 0.0

    def push_frame(self, frame):
        timestamp = tm.process_time()
        if len(self.rframes) == 0:
            # Dummy delta time for the first iteration
            delta_time = 1.0
        else:
            delta_time = timestamp - self.frames[self.curr_i-1][2]

        # Push it into the uniform sampling buffer
        if len(self.frames) < 2*self.ulen:
            self.frames.append((self.next_frame_id, frame, timestamp))
            self.curr_i += 1
        # Since the buffer is full resample every even frame, discard odd frame and
        # update the time threshold to make sure sampling is somewhat uniform
        elif self.curr_i == 2*self.ulen:
            self.curr_i = self.ulen
            for i in range(1, self.ulen):
                self.frames[i] = self.frames[i * 2]
            # Average the resampled time diff between each frame and use that to update the frame
            # sampling process
            self.time_thresh = sum([self.frames[i][2] - self.frames[i - 1][2] for i in range(1, self.ulen)])/(self.ulen - 1)
            self.frames[self.curr_i] = (self.next_frame_id, frame, timestamp)
            self.curr_i += 1
        elif delta_time > self.time_thresh:
            self.frames[self.curr_i] = (self.next_frame_id, frame, timestamp)
            self.curr_i += 1


        # Push it into the recent buffer
        if len(self.rframes) < self.rlen:
            self.rframes.append((self.next_frame_id, frame, timestamp))
        else:
            self.rframes.pop(0)
            self.rframes.append((self.next_frame_id, frame, timestamp))

        self.next_frame_id += 1
