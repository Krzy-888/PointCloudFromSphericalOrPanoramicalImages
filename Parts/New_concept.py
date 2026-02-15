import cv2
import numpy as np
from scipy.spatial import ConvexHull
import pointVis as pv
import matplotlib.pyplot as plt

# sphere image parameters
f = 1
pp = (0,0)

def imgxy2latlon(img,points):
    PI = np.pi
    H, W = img.shape
    transormationX = 2*PI/W
    transormationY = PI/H
    n_points = []
    # for p in points:
    #     x = p[0]
    #     y = p[1]
    lon = points[:,0]*transormationX - PI
    lat = PI/2 - points[:,1]*transormationY 
    # n_points.append([lat,lon])
    return lon, lat
    # return n_points
def sphrere3dpoints(lon,lat):
    x = np.cos(lat[:])*np.sin(lon[:])
    z = -np.sin(lat[:])
    y = np.cos(lat[:])*np.cos(lon[:])
    return x,y,z
def conv2deg(radians):
    degs = radians * 180/np.pi
    return degs
# img = np.zeros((180,360),dtype=np.uint8)
img = cv2.imread('Test-Dataset/VLC-2.png',0)
# points = np.array([
#     [20,20],
#     [120,100]
# ])
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img,None)
points = pv.convert_kp2xy(kp1)

plt.scatter(points[:,0],points[:,1],c='r')
plt.show()
plt.imshow(img)
plt.scatter(points[:,0],points[:,1],c='r')
plt.show()
lon, lat = imgxy2latlon(img,points)
lat_deg = conv2deg(lat)
lon_deg = conv2deg(lon)
plt.imshow(img, extent=[-180, 180, -90, 90])
plt.scatter(lon_deg,lat_deg)
plt.xlabel("longitude [deg]")
plt.ylabel("latitude [deg]")
# plt.gca().invert_yaxis()
plt.show()
fig = plt.figure()
ax = plt.axes(projection='3d')
x,y,z = sphrere3dpoints(lon,lat)
ax.scatter(x, y, z)
ax.scatter(0, 0, 0,c='r')
plt.show()