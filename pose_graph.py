import numpy as np
import cv2
import open3d as o3d
from copy import deepcopy
import pickle
from os import path
import json


class Node:
    def __init__(self, pcl_id, pcl, pose):
        self.pcl = pcl
        self.pcl_id = pcl_id
        self.pose = pose

class Edge:
    # n1 - Index of Node 1 in the list of nodes in the graph
    # n2 - Same as above but for Node 2
    # relative_transform - Transformation between n1 and n2
    def __init__(self, n1, n2, relative_transform):
        self.edge = (n1, n2)
        self.transform = relative_transform
        self.info_matrix = np.identity(4)

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

    def visualize(self, voxel_size=0.05):
        pcd = deepcopy(self.nodes[0].pcl)
        for n in self.nodes[1:]:
            pcd2 = deepcopy(n.pcl)
            pcd2.transform(n.pose)
            pcd += pcd2
        pcd.voxel_down_sample(voxel_size)
        o3d.visualization.draw_geometries([pcd])

    def store(self, fname):
        asset_dir = "assets"
        self.pickle_dict["nodes"] = []
        for n in self.nodes:
            fname_pcl = f"pcl_{n.pcl_id}.pcd"
            fname_pcl = path.join(asset_dir, fname_pcl)
            self.pickle_dict["nodes"].append((fname_pcl, n.pose, n.pcl_id))
            o3d.io.write_point_cloud(fname_pcl, n.pcl)
        self.pickle_dict["edges"] = self.edges
        pic = pickle.dumps(self.pickle_dict)
        with open(fname, "wb") as f:
            f.write(pic)

    
    @staticmethod
    def load(fname):
        d = None
        with open(fname, "rb") as f:
            d = pickle.load(f)
        pg = PoseGraph()
        print(d)
        for e in d["edges"]:
            pg.add_edge(e)

        for pcl_fname, pose, pcl_id in d["nodes"]:
            pcl = o3d.io.read_point_cloud(pcl_fname)
            pg.insert_node(Node(pcl_id, pcl, pose))
        print(pg)
        return pg
