#Data was gathered from: https://www.researchgate.net/publication/321048476_A_Post-Rectification_Approach_of_Depth_Images_of_Kinect_v2_for_3D_Reconstruction_of_Indoor_Scenes

import numpy as np

#In mm
FxIR = 388.198
FyIR = 389.033

#In Pixel Coordinates
DEPTH_WIDTH = 512
DEPTH_HEIGHT = 424
CxIR = DEPTH_WIDTH//2
CyIR = DEPTH_HEIGHT//2

#Distortion Coefficients
K1_IR = 0.126
K2_IR = -0.329
K3_IR = 0.111
P1_IR = -0.001
P2_IR = -0.002

#Skew
S_IR = 0

IR_DEPTHSHIFT = 60.358

RMat = np.array([[0.99997, 0.00715, -0.00105],
    [-0.00715, 0.99995, 0.00662],
    [0.00110, -0.00661, 0.99998]])
Tvec = np.array([-0.06015, 0.00221, 0.02714])

#KinvMat = np.array([[1/FxIR, 

def convert_from_uvd(u, v, d):
    d += IR_DEPTHSHIFT
    x_over_z = (CxIR - u)/FxIR
    y_over_z = (CyIR - v)/FyIR
    z = d / (1 + x_over_z**2 + y_over_z**2)
    x_uncorr = x_over_z 
    y_uncorr = y_over_z
    r2 = x_uncorr**2 + y_uncorr**2
    #x = x_uncorr
    #y = y_uncorr
    x = x_uncorr * (1 + K1_IR*r2 + K2_IR*r2**2 + K3_IR*r2**3) + 2*P1_IR*x_uncorr*y_uncorr + P2_IR*(r2 + 2*x_uncorr**2)
    x *= z
    y = y_uncorr * (1 + K1_IR*r2 + K2_IR*r2**2 + K3_IR*r2**3) + 2*P2_IR*x_uncorr*y_uncorr + P1_IR*(r2 + 2*y_uncorr**2)
    y *= z
    return x, y, z
