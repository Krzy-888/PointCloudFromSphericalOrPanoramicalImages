import cv2
import numpy as np
from scipy.spatial import ConvexHull
import pointVis as pv
import matplotlib.pyplot as plt

##   IMAGES IN BLOCK AND CAMERA CALIBRATION PARAMETERS
f = 1
pp = (0,0)
imgnumber = 5

##  FUNCTIONS
def imgxy2latlon(img,points):
    PI = np.pi
    H, W = img.shape
    transormationX = 2*PI/W
    transormationY = PI/H
    #     x = p[0]
    #     y = p[1]
    lon = points[:,0]*transormationX - PI
    lat = PI/2 - points[:,1]*transormationY 
    return lon, lat

def sphrere3dpoints(lon,lat):
    x = np.cos(lat[:])*np.sin(lon[:])
    y = -np.sin(lat[:])
    z = np.cos(lat[:])*np.cos(lon[:])
    return x,y,z

def conv2deg(radians):
    degs = radians * 180/np.pi
    return degs

def SetEqualScalesForAxis(ax,points):
    all_points = np.vstack(points)
    x_min, y_min, z_min = np.min(all_points, axis=0)
    x_max, y_max, z_max = np.max(all_points, axis=0)

    max_range = np.max([x_max - x_min, y_max - y_min, z_max - z_min]) / 2
    
    mid_x = (x_max + x_min) / 2
    mid_y = (y_max + y_min) / 2
    mid_z = (z_max + z_min) / 2

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_box_aspect([1,1,1])
def RotateAndShiftPoints(points,Rmatrix,Tvector):
    """
    Docstring for RotateAndShiftPoints

    :param points: Description
    :param Rmatrix: Description
    :param Tvector: Description
    """
    x1, y1, z1 = points
    points1_3d = np.vstack([x1, y1, z1]).T
    points1_3d = points1_3d @ Rmatrix.T
    points1_3d_shifted = points1_3d + Tvector
    return points1_3d_shifted

def RayFromPoints(A, B, length=100):
    """
    A, B : points [x,y,z])
    length: ray lenght
    """

    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)

    d = B - A

    # normalization
    d = d / np.linalg.norm(d)
    C = A + d * length
    return C

def VisualizeRay(ax, A, B, L=100):
    """
    ax: Axis
    A, B : points ([x,y,z])
    length: ray lenght
    """
    C = RayFromPoints(A,B,L)
    ax.plot([A[0], C[0]],
        [A[1], C[1]],
        [A[2], C[2]])

def FindCrossingPoints(StartPS1,PointOnS1,StartPS2,PointOnS2,tolearnce=1e-6):
    # ReyEnd1 = RayFromPoints(StartPS1,PointOnS1,ReyLenght)
    # ReyEnd2 = RayFromPoints(StartPS2,PointOnS2,ReyLenght)
    A = np.array(StartPS1, float)
    B = np.array(StartPS2, float)
    C = np.array(PointOnS1, float)
    D = np.array(PointOnS2, float)
    d = C - A
    e = D - B
    d /= np.linalg.norm(d)
    e /= np.linalg.norm(e)
    r = A - B
    a = np.dot(d, d)
    b = np.dot(d, e)
    c = np.dot(e, e)
    d0 = np.dot(d, r)
    e0 = np.dot(e, r)
    denom = a*c - b*b
    t = (b*e0 - c*d0) / denom
    s = (a*e0 - b*d0) / denom
    P1 = A + t*d
    P2 = B + s*e
    dist = np.linalg.norm(P1 - P2)
    if t >= 0 and s >= 0 and dist < tolearnce:
        return P1
    else:
        # print("Nothing Found")
        # return np.array([0,0,0])
        return P1
##  IMAGES READING
images = []
for i in range(imgnumber):
    img = cv2.imread(f'Test-Dataset/Campus/Campus_{i+1}.JPG',0)
    images.append(img)

##  POINT DETECTION
sift = cv2.SIFT_create()
kpoints = []
for i in range(imgnumber):
    kp, des = sift.detectAndCompute(images[i],None)
    points = pv.convert_kp2xy(kp)
    kpoints.append([kp,des,points])

##  POINT MACHING
mached_points_1 = []
mached_points_2 = []
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
for i in range(imgnumber):
    if i == imgnumber-1:
        print("Maching completed")
        continue
    else:
        des1 = kpoints[i][1]
        des2 = kpoints[i+1][1]
        kp1 = kpoints[i][0]
        kp2 = kpoints[i+1][0]
    p1m, p2m = pv.love_Test_filter(flann,des1,des2,kp1,kp2)
    mached_points_1.append(p1m)
    mached_points_2.append(p2m)

##  CALCULATING SPHERICAL COORDINAES FOR ALL MACHED POINTS 
spherical_mached_1 = []
spherical_mached_2 = []
for i in range(imgnumber):
    if i == imgnumber-1:
        print("CALCULATING SPHERICAL COORDINAES FOR ALL MACHED POINTS completed")
        continue
    else:
        lon_1, lat_1 = imgxy2latlon(images[i],kpoints[i][2])
        lat_deg_1 = conv2deg(lat_1)
        lon_deg_1 = conv2deg(lon_1)
        lon_2, lat_2 = imgxy2latlon(images[i+1],kpoints[i][2])
        lat_deg_2 = conv2deg(lat_2)
        lon_deg_2 = conv2deg(lon_2)
    spherical_mached_1.append([lon_1,lat_1])
    spherical_mached_2.append([lon_2,lat_2])

##  CALCULATING 3D SPHERICAL COORDINAES FOR ALL MACHED POINTS
spherical_3d_mached_1 = []
spherical_3d_mached_2 = []
for i in range(imgnumber):
    if i == imgnumber-1:
        print("CALCULATING 3D SPHERICAL COORDINAES FOR ALL MACHED POINTS completed")
        continue
    x1,y1,z1 = sphrere3dpoints(spherical_mached_1[i][0],spherical_mached_1[i][1])
    x2,y2,z2 = sphrere3dpoints(spherical_mached_2[i][0],spherical_mached_2[i][1])
    spherical_3d_mached_1.append([x1,y1,z1])
    spherical_3d_mached_2.append([x2,y2,z2])

## CALCULATION 3D RELATION TO ALL IMAGES IN BLOCK
cmaera_matrix = np.array(
    [[f,0,0],
     [0,f,0],
     [0,0,1]]
)
translationmatrixs = []
rotationmatrixs = [np.eye(3)]
for i in range(imgnumber):
    if i == imgnumber-1:
        print("CALCULATION 3D RELATION TO ALL IMAGES IN BLOCK completed")
        continue
    else:
        Essential_matrix,inliers = cv2.findEssentialMat(mached_points_1[i],mached_points_2[i],cmaera_matrix,cv2.RANSAC,threshold=3.0)
        points_3D, R, t, _ = cv2.recoverPose(
            Essential_matrix,
            mached_points_1[i],
            mached_points_2[i],
            mask= inliers
        )
        translationmatrixs.append(t.ravel())
        rotationmatrixs.append(R)

##  Calculation relation position for all images
positions = [[0,0,0]]
for i in range(imgnumber):
    if i == imgnumber-1:
        print("Calculation relation position for all images completed")
        continue
    else:
        next_pos = positions[i] + translationmatrixs[i]
        positions.append(next_pos)

## Calculation all points in model
all_new_model_points = []
duplicant_start_point = []

for i in range(imgnumber-1):
    if i == imgnumber-1:
        print("Calculation all points in model completed")
        continue
    else:
        points1_3d_shifted = RotateAndShiftPoints(spherical_3d_mached_1[i],rotationmatrixs[i],positions[i])
        all_new_model_points.append(points1_3d_shifted)
        duplicant_start_point.append(positions[i])

        points2_3d_shifted = RotateAndShiftPoints(spherical_3d_mached_1[i],rotationmatrixs[i+1],positions[i+1])
        all_new_model_points.append(points2_3d_shifted)
        duplicant_start_point.append(positions[i+1])

##  Visualize
fig = plt.figure()
ax = plt.axes(projection='3d')

for i, pts in enumerate(all_new_model_points):
    ax.scatter(pts[:,0], pts[:,1], pts[:,2], c='g')
    ax.scatter(duplicant_start_point[i][0], duplicant_start_point[i][1], duplicant_start_point[i][2], c='r')
SetEqualScalesForAxis(ax,all_new_model_points)
plt.show()

## Visualize 1st. mached homology rays
fig = plt.figure()
ax = plt.axes(projection='3d')

for i, pts in enumerate(all_new_model_points):
    ax.scatter(pts[0,0], pts[0,1], pts[0,2], c='g')
    ax.scatter(duplicant_start_point[i][0], duplicant_start_point[i][1], duplicant_start_point[i][2], c='r')
    VisualizeRay(ax,duplicant_start_point[i],pts[0,:],800)
SetEqualScalesForAxis(ax,all_new_model_points)
plt.show()

## Visualize 1st. mached homology points
# fig = plt.figure()
# ax = plt.axes(projection='3d')
point_cloud = []
for i in range(0,len(all_new_model_points),2):
    if i == len(all_new_model_points) - 1:
        continue
    POS = all_new_model_points[i][:,:]
    # POS2 = all_new_model_points[i+1][:,:]
    SSP1 = duplicant_start_point[i]
    SSP2 = duplicant_start_point[i+1]

    # ax.scatter(SSP1[0], SSP1[1], SSP1[2], c='r')
    # ax.scatter(SSP2[0], SSP2[1], SSP2[2], c='r')
    for j in range(len(POS)):
        POS1 = all_new_model_points[i][j,:]
        POS2 = all_new_model_points[i+1][j,:]
        CP = FindCrossingPoints(SSP1,POS1,SSP2,POS2)
        # ax.scatter(CP[0], CP[1], CP[2], c='b')
        point_cloud.append(np.array(CP))
# plt.show()
np.savetxt("points2.csv", point_cloud, delimiter=",",)
