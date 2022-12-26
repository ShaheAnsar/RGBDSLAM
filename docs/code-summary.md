# Code Summary

### Internal Math

This file implements necessary mathematical functions. It contains
an implementation of the Umeyama orientation algorithm (Originally written
by someone else). It also has a function to compute pairwise squared distances
between any three 3D points. There's another function that uses a C library binding
to implement the same thing with RANSAC for performance reasons. Originally this
was done entirely in Python but was too slow.

### Kinect Preprocessing

This file implements some preprocessing functions to convert/clean data
coming from the Kinect V2 sensor.

* convert\_rgb\_depth\_to\_pcl  

This function takes in an RGBD image and converts it into a point cloud. It also
gets rid of invalid points and distortion.

* get\_depth  

This function takes in undistorted depth and interpolates for 
non integer indices. A relic of previous versions, not used anymore

* get\_xyz  

Used to convert pixel positions to real world 3D points

### Save Frames  

This script saves an RGBD frame as a separate RGB frame and a depth buffer.
The images are registered and undistorted.

### Save Kinect Params  

This script saves various intrinsic camera parameters used for mathematical functions.  

### Play Frames  

Used to play frames that are part of the Mobile RGBD dataset. Used for simulation purposes.  

### Mobrgbd  

A module used to get and manage frames and data from the Mobile RGBD Dataset  

### View Map  

A script used to display the map of the environment for any video from the MobileRGBD
dataset.  

### Pose Graph  

Implements a pose graph that can be further used for optimization purposes  

### Frame Listener  

Used to listen and store frames from the Freenect2 library  

### Frame Sampler  

Used to sample frames and store them for comparison. There are two sets - 12 uniformly sampled frames, 3 recent frames.
Every frame is compared to these, the strongest fit for each set is returned. The recent frames set is used for the connecting
edges between every node and the uniformly sampled set is mainly used to for loop closures.  
