"""
## References
- [Umeyama's paper](http://edge.cs.drexel.edu/Dmitriy/Matching_and_Metrics/Umeyama/um.pdf)
- [CarloNicolini's python implementation](https://gist.github.com/CarloNicolini/7118015)
"""
from __future__ import print_function
import numpy as np
import open3d as o3d

def similarity_transform(from_points, to_points):

    assert len(from_points.shape) == 2, \
            "from_points must be a m x n array"
    assert from_points.shape == to_points.shape, \
        "from_points and to_points must have the same shape"

    N, m = from_points.shape

    mean_from = from_points.mean(axis=0)
    mean_to = to_points.mean(axis=0)

    delta_from = from_points - mean_from  # N x m
    delta_to = to_points - mean_to       # N x m

    sigma_from = (delta_from * delta_from).sum(axis=1).mean()
    sigma_to = (delta_to * delta_to).sum(axis=1).mean()

    cov_matrix = delta_to.T.dot(delta_from) / N

    U, d, V_t = np.linalg.svd(cov_matrix, full_matrices=True)
    cov_rank = np.linalg.matrix_rank(cov_matrix)
    S = np.eye(m)

    if cov_rank >= m - 1 and np.linalg.det(cov_matrix) < 0:
        S[m-1, m-1] = -1
    elif cov_rank < m-1:
        raise ValueError("colinearility detected in covariance matrix:\n{}".format(cov_matrix))

    R = U.dot(S).dot(V_t)
    c = (d * S.diagonal()).sum() / sigma_from
    t = mean_to - c*R.dot(mean_from)

    return c*R, t

# Takes in an array of 3D vectors and calculates the pairwise distance between every pair of vectors
def get_pairwise_dist2(pts):
    dists = []
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            v = pts[i] - pts[j]
            dists.append(np.dot(v,v))
    return np.array(dists)

#def similarity_transform_o3d(pcd1, pcd1feat, pcd2, pcd2feat, distance_thresh, ransac_iter=100000, ransac_p = 0.999):
#    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(pcd1, pcd2, pcd1feat, pcd2feat, True, distance_thresh,
#            o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4, [
#                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
#                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_thresh)],
#            o3d.pipelines.registration.RANSACConvergenceCriteria(ransac_iter, ransac_p))
#    return result
def similarity_transform_o3d_rough(pcd1, pcd1feat, pcd2, pcd2feat, distance_thresh):
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
            pcd1, pcd2, pcd1feat, pcd2feat, o3d.pipelines.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=distance_thresh))
    return result

def similarity_transform_o3d_precise(pcd1, pcd2, distance_thresh, init_trans):
    crit = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    result = None
    try:
        result = o3d.pipelines.registration.registration_colored_icp(pcd1, pcd2, distance_thresh, init_trans, o3d.pipelines.registration.TransformationEstimationForColoredICP(), crit)
    except Exception as e:
        pass
    return result
