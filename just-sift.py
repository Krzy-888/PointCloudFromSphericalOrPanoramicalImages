import cv2
import matplotlib.pyplot as plt
import numpy as np
import pointVis as pv
# from matplotlib.collections import LineCollection

print("test")
img_1 = cv2.imread("Test-Dataset/VLC-1.png",0)
img_2 = cv2.imread("Test-Dataset/VLC-2.png",0)
# plt.imshow(img_1)
# plt.show()
# plt.imshow(img_2)
# plt.show()
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img_1,None)
kp2, des2 = sift.detectAndCompute(img_2,None)
p1 = pv.convert_kp2xy(kp1)
print(len(p1))
plt.imshow(img_1,cmap="gray")
pv.show_points(plt,p1)
plt.show()

p2 = pv.convert_kp2xy(kp2)
print(len(p2))
plt.imshow(img_2,cmap="gray")
pv.show_points(plt,p2)
plt.show()

#Filter and mach
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

p1m, p2m = pv.love_Test_filter(flann,des1,des2,kp1,kp2)

# pv.show_maches_in_axis(plt,img_1,img_2,p1m,p2m,'g')
# plt.show()
H, mask = cv2.findHomography(
    p1m, p2m,
    cv2.RANSAC,
    5.0
)
print(H)
pts1, pts2 = pv.filter_RANSAC(p1m,p2m,mask)
pv.show_maches_in_axis(plt,img_1,img_2,pts1,pts2,'g')
plt.show()