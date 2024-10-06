# RGBDSLAM
This is an implementation of RGBD SLAM. It uses a Kinect V2 to acquire the RGBD images. These images are then converted to point clouds. Two sets of point clouds are maintained - recent ones (to connect the next incoming point cloud)
and a uniformly sampled set of pointclouds since t = 0. The latter is used to detect loop closures. This can be improved if I am willing to store point clouds and dynamically stream them into memory based on my current estimated location.
Registration of the point clouds against each other is done in two phases - rough and precise. The rough phase will take massively differently oriented point clouds and get them somewhat close. The precise step will refine it further to
make the similarity transform accurate. This transform is then used to connect one pose to another. This allows us to understand how the robot and the environment moves. Loop closure is tricky, since we need to understand if the two point clouds
actually are the same thing or not. This can be tricky, but it is safer to err on the side of not having false loop closures, which is what we do. The thresholds need to be manually set for the environment right now. The point clouds are also matched
using two different techniques - one uses FPFH descriptors to match point clouds directly. The other one uses image features (SIFT, SURF or ORB are all suitable), removes outliers with RANSAC and then uses the depth buffer to find out the similarity
transform. This is done by finding a set of 3 features, forming a triangle and seeing how it transforms from one coordinate frame to another.  
Once we have this graph ready, we construct a factor graph for GTSAM. GTSAM then optimizes over it. We then use these optimized values to orient the point clouds and build the map.
