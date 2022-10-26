import numpy as np
import cv2

filename = "../../color/MobileRGBD/Corridor/Traj_54_-15_Corridor_0.3/video/MobileRGBD/Corridor/Traj_54_-15_Corridor_0.3/video/video.raw"

w = 1920
h = 1080

f = open(filename, "rb")
def get_frame():
    bslice = bytearray(f.read(w*h*2))
    frame = np.frombuffer(bslice, dtype=np.uint8)
    frame = frame.reshape((h, w, 2))
    frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUY2)
    return frame


frame1 = get_frame()
frame1 = cv2.resize(frame1, (0, 0), fx=0.5, fy=0.5)
frame2 = get_frame()
frame2 = cv2.resize(frame2, (0, 0), fx=0.5, fy=0.5)

sift = cv2.xfeatures2d.SIFT_create()
#surf = cv2.xfeatures2d.SURF_create()
orb = cv2.ORB_create(nfeatures=1500)
ksift, dsift = sift.detectAndCompute(frame1, None)
ksift2, dsift2 = sift.detectAndCompute(frame2, None)
#ksurf, _ = surf.detectAndCompute(frame1, None)
korb, dorb = orb.detectAndCompute(frame1, None)

matcher_sift = cv2.BFMatcher(crossCheck=True)
matcher_orb = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = matcher_sift.match(dsift, dsift2)
matches = sorted(matches, key = lambda x: x.distance)
match_img = cv2.drawMatches(frame1, ksift, frame2, ksift2, matches[:30], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
imgsift = cv2.drawKeypoints(frame1, ksift, None)
imgsift2 = cv2.drawKeypoints(frame2, ksift2, None)
#imgsurf = cv2.drawKeypoints(frame1, ksurf, None)
imgorb = cv2.drawKeypoints(frame1, korb, None)

cv2.imshow("SIFT", imgsift)
cv2.imshow("SIFT2", imgsift2)
#cv2.imshow("SURF", imgsurf)
cv2.imshow("ORB", imgorb)
cv2.imshow("Mathc", match_img)
cv2.waitKey(0)
