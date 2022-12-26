# Planned Future Changes

* Change the backend from GTSAM/G2O to SE-Sunc or Ceres
* Implement the majority of the code in Julia to improve speed
* Replace RANSAC with PROSAC to improve performace
* Have a custom implementation of FPFH feature detection for the above
* Add an IMU to improve performance even further
* Use multithreading and/or CUDA/OpenCL to improve performance even further
* Calibrate each Kinect for different emissivities and depths
