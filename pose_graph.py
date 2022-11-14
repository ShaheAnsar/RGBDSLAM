import numpy as np
import cv2
import open3d as o3d


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
    def __init__(self):
        self.nodes = []
        self.edges = []

    def insert_node(self, n):
        self.nodes.append(n)

    def add_edge(self, edge):
        self.edges.append(edge)

    def exists_node(self, pcl_id):
        for i in self.nodes:
            if i.pcl_id == pcl_id:
                return True
        return False
