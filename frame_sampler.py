import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import cv2
import time as tm
import freenect2 as fr2
import json
import internal_math as im
from copy import deepcopy, copy
from time import perf_counter


#Maintains 15 frames, of which 12 are uniformly sampled and the remainig 3 are the most recent ones
class FrameSampler:
    def __init__(self, dev, uni_len=12, rec_len=3 ):
        self.frames = []
        self.rframes = []
        self.next_frame_id = 0
        self.ulen = uni_len
        self.rlen = rec_len
        self.dev = dev
        # Used to denote the index of the last frame appended
        self.curr_i = 0
        # Time diff used to sample frames
        self.time_thresh = 0.0
        # Voxel size for downsampling
        self.VOXEL_SIZE = 0.1
        # Distance thresholding
        self.DIST_THRESH = 0.02

    def push_frame(self, frame_set):
        timestamp = tm.process_time()
        if len(self.rframes) == 0:
            # Dummy delta time for the first iteration
            delta_time = 1.0
        else:
            delta_time = timestamp - self.frames[self.curr_i-1][2]
        
        pcl = self.__preprocess_frame(frame_set)

        # Push it into the uniform sampling buffer
        if len(self.frames) < 2*self.ulen:
            self.frames.append((self.next_frame_id, pcl, timestamp))
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
            if delta_time > self.time_thresh:
                self.frames[self.curr_i] = (self.next_frame_id, pcl, timestamp)
                self.curr_i += 1
        elif delta_time > self.time_thresh:
            self.frames[self.curr_i] = (self.next_frame_id, pcl, timestamp)
            self.curr_i += 1


        # Push it into the recent buffer
        if len(self.rframes) < self.rlen:
            self.rframes.append((self.next_frame_id, pcl, timestamp))
        else:
            self.rframes.pop(0)
            self.rframes.append((self.next_frame_id, pcl, timestamp))

        self.next_frame_id += 1

    def __preprocess_frame(self, frame_set):
        depth_undistorted, rgb_registered = self.dev.registration.apply(frame_set[0], frame_set[1], enable_filter=True)
        depth_undistorted = deepcopy(depth_undistorted.to_array())
        rgb_registered = deepcopy(rgb_registered.to_array())
        o3d_rgb = o3d.geometry.Image(cv2.cvtColor(rgb_registered, cv2.COLOR_BGR2RGB))
        o3d_depth = o3d.geometry.Image(depth_undistorted)
        o3d_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_rgb, o3d_depth, convert_rgb_to_intensity=False, depth_trunc=5.0)
        ir_params = self.dev.ir_camera_params
        o3d_pcl = o3d.geometry.PointCloud.create_from_rgbd_image(o3d_rgbd, o3d.camera.PinholeCameraIntrinsic(512, 424, ir_params.fx, ir_params.fx, ir_params.cx, ir_params.cy))
        o3d_pcl, o3d_feat = self.__preprocess_pcl(o3d_pcl)
        return o3d_pcl, o3d_feat
    
    def __preprocess_pcl(self, pcl):
        down = pcl.voxel_down_sample(self.VOXEL_SIZE)
        down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=self.VOXEL_SIZE*2,max_nn=30))
        down_features = o3d.pipelines.registration.compute_fpfh_feature(down, o3d.geometry.KDTreeSearchParamHybrid(radius=self.VOXEL_SIZE*5, max_nn=100))
        return down, down_features

    def compare_and_push_frame(self, frame_set):
        pcd1, pcd1feat = self.__preprocess_frame(frame_set)
        best_rec = (0, 0, 0)
        for i in reversed(self.rframes):
            res = im.similarity_transform_o3d(pcd1, pcd1feat, i[1][0], i[1][1], self.DIST_THRESH)
            if best_rec[-1] == 0 or res.fitness > best_rec[-1].fitness:
                best_rec = (i[0], i[1][0], res)
        best_uni = (0, 0, 0) 
        for i in self.frames[:self.ulen]:
            res = im.similarity_transform_o3d(pcd1, pcd1feat, i[1][0], i[1][1], self.DIST_THRESH)
            if best_uni[-1] == 0 or res.fitness > best_uni[-1].fitness:
                best_uni = (i[0], i[1][0], res)
        self.push_frame(frame_set)
        # Point Cloud data, Point Cloud ID, Best Recent Comp, Best Uniform Comp
        return (pcd1, self.rframes[-1][0], best_rec, best_uni)
