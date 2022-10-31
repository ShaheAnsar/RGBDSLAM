# This script matches features and filters out rubbish


from freenect2 import Device, FrameType
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
from os import path
import math as m
import random as r
from time import perf_counter
import kinect_preprocess as p
import internal_math as im
import json


ASSET_DIR = "assets"
FR1_NAME = "fr1"
FR2_NAME = "fr2"
PARAMS_NAME = "params.json"
FR1_PATH = path.join(ASSET_DIR, FR1_NAME)
FR2_PATH = path.join(ASSET_DIR, FR2_NAME)
dev = Device()
dev.start()
dev.close()

# Import the data
fr1_rgb = cv2.imread(FR1_PATH + "_rgb.png")
fr2_rgb = cv2.imread(FR2_PATH + "_rgb.png")
fr1_depth = np.load(FR1_PATH + "_depth.npy")
fr2_depth = np.load(FR2_PATH + "_depth.npy")
f = open(path.join(ASSET_DIR, PARAMS_NAME))
params = json.load(f)
f.close()
ir_params = params["IR"]
rgb_params = params["COLOR"]

orb = cv2.ORB_create(nfeatures=500)
kp, des = orb.detectAndCompute(fr1_rgb, None)
kp2, des2 = orb.detectAndCompute(fr2_rgb, None)
fr1_keypoints = np.array([list(i.pt) for i in kp])
fr2_keypoints = np.array([list(i.pt) for i in kp])
print(fr1_keypoints)
fr1_detected = cv2.drawKeypoints(fr1_rgb, kp, None, color=(0,255,0), flags=0)
fr2_detected = cv2.drawKeypoints(fr2_rgb, kp2, None, color=(0,255,0), flags=0)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#cv2.imshow("ORB", fr1_detected)
#cv2.imshow("ORB2", fr2_detected)
# Feature matching
#print(timeit.timeit("bf.match(des, des2)", globals=globals(), number=2))
matches = bf.match(des, des2)
# Used to get rid of invalid entries from the matches
def pts_from_match(x):
    global fr1_depth, fr1_keypoints, ir_params
    fr1_index = x.queryIdx
    fr2_index = x.trainIdx
    pt1 = p.get_xyz(fr1_depth, fr1_keypoints[fr1_index, 1], fr1_keypoints[fr1_index, 0], ir_params)
    pt2 = p.get_xyz(fr2_depth, fr2_keypoints[fr2_index, 1], fr2_keypoints[fr2_index, 0], ir_params)
    return (pt1, pt2)
with open("log.txt", "w") as f:
    for i in matches:
        f.write(f"{pts_from_match(i)}\n")

def filter_matches(x):
    global fr1_depth, fr1_keypoints, ir_params
    fr1_index = x.queryIdx
    fr2_index = x.trainIdx
    pt1 = p.get_xyz(fr1_depth, fr1_keypoints[fr1_index, 1], fr1_keypoints[fr1_index, 0], ir_params)
    pt2 = p.get_xyz(fr2_depth, fr2_keypoints[fr2_index, 1], fr2_keypoints[fr2_index, 0], ir_params)
    if np.any(np.isnan(pt1)) or np.any(np.isnan(pt2)) or np.any(np.abs(pt1) < 1.0) or np.any(np.abs(pt2) < 1.0) or np.abs(pt1[2]) < 10.0 or np.abs(pt2[2]) < 10.0:
        return False
    return True
matches = [i for i in matches if filter_matches(i)]
print(len(matches))
#matches = sorted(matches, key = lambda x : x.distance)
inliers = []
T = None
Ts = []
visited_dict1 = {}
visited_dict2 = {}
i1 = 0
i2 = 0
while True:
    if i1 == 30:
        break
    samples = r.choices(matches, k = 3)
    if tuple( samples ) in visited_dict1:
        continue
    else:
        visited_dict1[tuple( samples )] = True
    pts1 = []
    for i in samples:
        pts1.append(p.get_xyz(fr1_depth, fr1_keypoints[i.queryIdx, 1], fr1_keypoints[i.queryIdx, 0], ir_params))
    pts2 = []
    for i in samples:
        pts2.append(p.get_xyz(fr2_depth, fr2_keypoints[i.trainIdx, 1], fr2_keypoints[i.trainIdx, 0], ir_params))
    pts1 = np.array(pts1)
    pts2 = np.array(pts2)
    dists1 = im.get_pairwise_dist2(pts1)
    dists2 = im.get_pairwise_dist2(pts2)
    invalid = False
    for i in range(len(dists1)):
        if np.abs(dists1[i] - dists2[i]) >= 1000:
            invalid = True
    if invalid or np.any(dists1 == 0):
        continue
    i1 += 1
    T = im.similarity_transform(pts1, pts2)
    print(T)
    consensus = []
    while True:
        if i2 == 400:
            break
        samples = r.choices(matches, k = 3)
        if tuple(samples) in visited_dict2:
            continue
        else:
            visited_dict2[tuple(samples)] = True
        pts1 = []
        for i in samples:
            pts1.append(p.get_xyz(fr1_depth, fr1_keypoints[i.queryIdx, 1], fr1_keypoints[i.queryIdx, 0], ir_params))
        pts2 = []
        for i in samples:
            pts2.append(p.get_xyz(fr2_depth, fr2_keypoints[i.trainIdx, 1], fr2_keypoints[i.trainIdx, 0], ir_params))
        pts1 = np.array(pts1)
        pts2 = np.array(pts2)
        dists1 = im.get_pairwise_dist2(pts1)
        dists2 = im.get_pairwise_dist2(pts2)
        invalid = False
        for i in range(len(dists1)):
            if np.abs(dists1[i] - dists2[i]) >= 1000:
                invalid = True
        if invalid or np.any(dists1 == 0):
            continue
        i2 += 1
        pts_transformed = np.zeros(pts1.shape)
        for i in range(pts1.shape[0]):
            pts_transformed[i, :] = np.dot(T[0], pts1[i, :]) + T[1]
        delta = pts2 - pts_transformed
        error = np.sqrt(np.sum(np.square(delta)))
        threshold = 20
        if error < threshold:
            print(f"Transformed: {pts_transformed}\nPTS2: {pts2}")
            print(error)
            consensus.append(samples)
    Ts.append([T, consensus])

Ts = sorted(Ts, key=lambda x: len(x[1]), reverse=True)
print(Ts[:5])


#print(matches)
#print(np.random.choice(np.arange(0, fr1_keypoints.shape[0]), size=3, replace=False))

#cv2.waitKey(0)
dev.close()
print("Hello")
