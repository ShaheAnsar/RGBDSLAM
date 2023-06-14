import numpy as np
import cv2
import open3d as o3d
from copy import deepcopy
import pickle
from os import path
import json
import transforms3d as t3d
import gtsam as gs

class Node:
    def __init__(self, pcl_id, pcl, pose, opcl):
        self.pcl = pcl
        self.opcl = opcl
        self.pcl_id = pcl_id
        self.pose = pose
        T, R, Z, S = t3d.affines.decompose44(pose)
        self.T, self.R, self.Z, self.S = T, R, Z, S

    def __repr__(self):
        return f"PCL ID: {self.pcl_id}, Pose: {self.pose}, Optimized Pose: {self.oPose}"

    def optimized_pose(self, oT, oR):
        self.oPose = t3d.affines.compose(oT, oR, np.ones(3))

class Edge:
    ETYPE_NORMAL = 1
    ETYPE_LOOP = 2
    ETYPE_BROKEN = 3
    # n1 - Index of Node 1 in the list of nodes in the graph
    # n2 - Same as above but for Node 2
    # relative_transform - Transformation between n1 and n2
    def __init__(self, n1, n2, relative_transform, stats, etype=ETYPE_NORMAL):
        self.edge = (n1, n2)
        self.transform = relative_transform
        self.etype = etype
        self.stats = stats
        self.decompose()

    def decompose(self):
        T, R, Z, S = t3d.affines.decompose44(self.transform)
        self.T = T
        self.R = R
        self.S = S
        return T, R, S

class PoseGraph:
    def default_err(cls, pcd1, pcd2):
        return 1.0

    def __init__(self, error_fn=None):
        self.nodes = []
        self.edges = []
        if error_fn is None:
            error_fn = PoseGraph.default_err
        self.pickle_dict = {}


    def insert_node(self, n):
        self.nodes.append(n)

    def add_edge(self, edge):
        self.edges.append(edge)

    def exists_node(self, pcl_id):
        for i in self.nodes:
            if i.pcl_id == pcl_id:
                return True
        return False

    def get_node(self, pcl_id):
        for i in self.nodes:
            if i.pcl_id == pcl_id:
                return i
        raise RuntimeError(f"Node doesn't exist, PCL ID: {pcl_id}")

    def get_node_and_index(self, pcl_id):
        for i, n in enumerate(self.nodes):
            if n.pcl_id == pcl_id:
                return i, n

        raise RuntimeError(f"Node doesn't exist, PCL ID: {pcl_id}")

    def visualize(self, voxel_size=0.05, uniform_color = False):
        pcd = deepcopy(self.nodes[0].pcl)
        for n in self.nodes[1:]:
            pcd2 = deepcopy(n.pcl)
            pcd2.transform(n.pose)
            pcd += pcd2
        pcd = pcd.voxel_down_sample(voxel_size)
        if uniform_color:
            pcd.paint_uniform_color([0.3, 0.3, 0.3])
        o3d.visualization.draw_geometries([pcd])

    def optimized_visualize(self, voxel_size=0.05, uniform_color = False, full_res = False):
        pcd = None
        if full_res:
            pcd = deepcopy(self.nodes[0].opcl)
        else:
            pcd = deepcopy(self.nodes[0].pcl)
        #pcd = pcd.voxel_down_sample(voxel_size)
        for n in self.nodes[1:]:
            pcd2 = None
            if full_res:
                pcd2 = deepcopy(n.opcl)
                pcd2 = pcd2.voxel_down_sample(voxel_size)
            else:
                pcd2 = deepcopy(n.pcl)
            pcd2.transform(n.oPose)
            pcd += pcd2
        pcd = pcd.voxel_down_sample(voxel_size)
        pcd = pcd.remove_duplicated_points()
        if uniform_color:
            pcd.paint_uniform_color([0.3, 0.3, 0.3])
        o3d.visualization.draw_geometries([pcd])
        return pcd

    def store(self, fname):
        asset_dir = "assets"
        self.pickle_dict["nodes"] = []
        for n in self.nodes:
            fname_pcl = f"pcl_{n.pcl_id}.pcd"
            fname_opcl = f"opcl_{n.pcl_id}.pcd"
            fname_pcl = path.join(asset_dir, fname_pcl)
            fname_opcl = path.join(asset_dir, fname_opcl)
            self.pickle_dict["nodes"].append((fname_pcl, fname_opcl, n.pose, n.pcl_id))
            o3d.io.write_point_cloud(fname_pcl, n.pcl)
            o3d.io.write_point_cloud(fname_opcl, n.opcl)
        self.pickle_dict["edges"] = self.edges
        pic = pickle.dumps(self.pickle_dict)
        with open(fname, "wb") as f:
            f.write(pic)

    def construct_factor_graph(self):
        graph = gs.NonlinearFactorGraph()
        prior_noise = gs.noiseModel.Diagonal.Sigmas(np.ones(6)*0.03)
        odometry_noise_low = gs.noiseModel.Diagonal.Sigmas(np.ones(6)*0.03)
        loop_noise = gs.noiseModel.Diagonal.Sigmas(np.ones(6)*1000)
        broken_noise = gs.noiseModel.Diagonal.Sigmas(np.ones(6)*100000000)
        init_rot = gs.Rot3(np.identity(3))
        graph.add(gs.PriorFactorPose3(1, gs.Pose3(init_rot, np.zeros(3)), prior_noise))
        for e in self.edges:
            if e.etype == Edge.ETYPE_NORMAL:
                noise_model = gs.noiseModel.Diagonal.Sigmas(np.ones(6)*e.stats[2])
                graph.add(gs.BetweenFactorPose3(e.edge[0] + 1, e.edge[1] + 1, gs.Pose3(gs.Rot3( e.R ), e.T), noise_model))
            elif e.etype == Edge.ETYPE_LOOP:
                noise_model = gs.noiseModel.Diagonal.Sigmas(np.ones(6)*e.stats[2] * 20)
                graph.add(gs.BetweenFactorPose3(e.edge[0] + 1, e.edge[1] + 1, gs.Pose3(gs.Rot3( e.R ), e.T), noise_model))
            else:
                graph.add(gs.BetweenFactorPose3(e.edge[0] + 1, e.edge[1] + 1, gs.Pose3(gs.Rot3( e.R ), e.T), broken_noise))

        self.fgraph = graph

    def optimize(self):
        initial_estimate = gs.Values()
        for i in range(0, len(self.nodes)):
            initial_estimate.insert(i + 1, gs.Pose3(gs.Rot3(self.nodes[i].R), self.nodes[i].T))
        optimizer = gs.LevenbergMarquardtOptimizer(self.fgraph, initial_estimate)
        result = optimizer.optimize()
        for i in range(0, len(self.nodes)):
            self.nodes[i].optimized_pose(result.atPose3(i + 1).translation(), result.atPose3(i + 1).rotation().matrix())
        return result
    
    # Visualize every edge with the transform applied
    def visualize_edges(self, uniform_color = True):
        for i, e in enumerate( self.edges ):
            n1 = self.nodes[e.edge[0]]
            n2 = self.nodes[e.edge[1]]
            pcd1 = deepcopy(n1.pcl)
            pcd2 = deepcopy(n2.pcl)
            if uniform_color:
                pcd1.paint_uniform_color([1.0, 0.0, 0.0])
                pcd2.paint_uniform_color([0.0, 1.0, 0.0])
            pcd2.transform(e.transform)
            o3d.visualization.draw_geometries([pcd1, pcd2])
            print(f"Edge {i}, Type: {e.etype}, Stats {e.stats}")
            cv2.waitKey(0)
            

    
    @staticmethod
    def load(fname):
        d = None
        with open(fname, "rb") as f:
            d = pickle.load(f)
        pg = PoseGraph()
        for e in d["edges"]:
            pg.add_edge(e)

        for pcl_fname, opcl_fname, pose, pcl_id in d["nodes"]:
            pcl = o3d.io.read_point_cloud(pcl_fname)
            opcl = o3d.io.read_point_cloud(opcl_fname)
            pg.insert_node(Node(pcl_id, pcl, pose, opcl))
        return pg
