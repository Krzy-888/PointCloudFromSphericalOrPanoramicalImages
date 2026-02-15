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
    y = -np.sin(lat[:])
    z = np.cos(lat[:])*np.cos(lon[:])
    return x,y,z
def conv2deg(radians):
    degs = radians * 180/np.pi
    return degs



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

## CALCULATING SPHERICAL COORDINAES FOR ALL POINTS 
spherical_kp = []
for i in range(imgnumber):
    lon, lat = imgxy2latlon(images[i],kpoints[i][2])
    lat_deg = conv2deg(lat)
    lon_deg = conv2deg(lon)
    # plt.imshow(img, cmap='gray',extent=[-180, 180, -90, 90])
    # plt.scatter(lon_deg,lat_deg)
    # plt.xlabel("longitude [deg]")
    # plt.ylabel("latitude [deg]")
    # plt.show()
    spherical_kp.append([lon,lat])

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
        # plt.imshow(images[i], cmap='gray',extent=[-180, 180, -90, 90])
        # plt.scatter(lon_deg_1,lat_deg_1)
        # plt.xlabel("longitude [deg]")
        # plt.ylabel("latitude [deg]")
        # plt.show()
        lon_2, lat_2 = imgxy2latlon(images[i+1],kpoints[i][2])
        lat_deg_2 = conv2deg(lat_2)
        lon_deg_2 = conv2deg(lon_2)
        # plt.imshow(images[i+1], cmap='gray',extent=[-180, 180, -90, 90])
        # plt.scatter(lon_deg_2,lat_deg_2)
        # plt.xlabel("longitude [deg]")
        # plt.ylabel("latitude [deg]")
        # plt.show()
    spherical_mached_1.append([lon_1,lat_1])
    spherical_mached_2.append([lon_2,lat_2])

##  CALCULATING 3D SPHERICAL COORDINAES FOR ALL POINTS
# fig = plt.figure()
# ax = plt.axes(projection='3d')
for i in range(imgnumber):
    
    x,y,z = sphrere3dpoints(spherical_kp[i][0],spherical_kp[i][1])
    # ax.scatter(x, y, z)
    # ax.scatter(0, 0, 0,c='r')
    # ax.set_box_aspect([1, 1, 1])
    # plt.show()

##  CALCULATING 3D SPHERICAL COORDINAES FOR ALL MACHED POINTS
spherical_3d_mached_1 = []
spherical_3d_mached_2 = []
# fig = plt.figure()
# ax = plt.axes(projection='3d')
for i in range(imgnumber):
    if i == imgnumber-1:
        print("CALCULATING 3D SPHERICAL COORDINAES FOR ALL MACHED POINTS completed")
        continue
    
    x1,y1,z1 = sphrere3dpoints(spherical_mached_1[i][0],spherical_mached_1[i][1])
    # ax.scatter(x1, y1, z1)
    # ax.scatter(0, 0, 0,c='r')
    # ax.set_box_aspect([1, 1, 1])
    # plt.show()
    print(len(x1))
    x2,y2,z2 = sphrere3dpoints(spherical_mached_2[i][0],spherical_mached_2[i][1])
    # ax.scatter(x2, y2, z2)
    # ax.scatter(0, 0, 0,c='r')
    # ax.set_box_aspect([1, 1, 1])
    # plt.show()
    # print(len(x2))
    spherical_3d_mached_1.append([x1,y1,z1])
    spherical_3d_mached_2.append([x2,y2,z2])
print(len(spherical_3d_mached_1))
print(len(spherical_3d_mached_2))
## CALCULATION 3D RELATION TO ALL IMAGES I BLOCK
cmaera_matrix = np.array(
    [[f,0,0],
     [0,f,0],
     [0,0,1]]
)
translationmatrixs = []
rotationmatrixs = [np.eye(3)]
for i in range(imgnumber):
    if i == imgnumber-1:
        continue
    else:
        Essential_matrix,inliers = cv2.findEssentialMat(mached_points_1[i],mached_points_2[i],cmaera_matrix,cv2.RANSAC,threshold=3.0)
        # points_3D, Rotation_mx,translation_mx = cv2.recoverPose(mached_points_1[i],mached_points_2[i],cmaera_matrix,E=Essential_matrix)
        points_3D, R, t, _ = cv2.recoverPose(
            Essential_matrix,
            mached_points_1[i],
            mached_points_2[i],
            mask= inliers
        )
        # print(points_3D, R,t, _)
        translationmatrixs.append(t.ravel())
        rotationmatrixs.append(R)
##  Calculation relation position for all images
positions = [[0,0,0]]
for i in range(imgnumber):
    if i == imgnumber-1:
        continue
    else:
        next_pos = positions[i] + translationmatrixs[i]
        positions.append(next_pos)

## Calculation all points in model
all_new_model_points = []
duplicant_start_point = []

for i in range(imgnumber-1):
    # pierwszy zestaw punktów
    if i == imgnumber-1:
        continue
    else:
    # print(positions[i])
        x1, y1, z1 = spherical_3d_mached_1[i]
        points1_3d = np.vstack([x1, y1, z1]).T
        points1_3d = points1_3d @ rotationmatrixs[i].T
        points1_3d_shifted = points1_3d + positions[i]
        all_new_model_points.append(points1_3d_shifted)
        duplicant_start_point.append(positions[i])

        # drugi zestaw punktów
        x2, y2, z2 = spherical_3d_mached_2[i]
        points2_3d = np.vstack([x2, y2, z2]).T
        points2_3d = points2_3d @ rotationmatrixs[i+1].T
        points2_3d_shifted = points2_3d + positions[i+1].T
        all_new_model_points.append(points2_3d_shifted)
        duplicant_start_point.append(positions[i+1])

##  Visualize
fig = plt.figure()
ax = plt.axes(projection='3d')

# Dodaj punkty
for i, pts in enumerate(all_new_model_points):
    ax.scatter(pts[:,0], pts[:,1], pts[:,2], c='g')
    ax.scatter(duplicant_start_point[i][0], duplicant_start_point[i][1], duplicant_start_point[i][2], c='r')

# Wyznacz wspólny zakres dla wszystkich osi
all_points = np.vstack(all_new_model_points)  # scalamy wszystkie punkty w jedną tablicę
x_min, y_min, z_min = np.min(all_points, axis=0)
x_max, y_max, z_max = np.max(all_points, axis=0)

# znajdź maksymalny rozmiar w każdej osi
max_range = np.max([x_max - x_min, y_max - y_min, z_max - z_min]) / 2

# wyznacz środek chmury
mid_x = (x_max + x_min) / 2
mid_y = (y_max + y_min) / 2
mid_z = (z_max + z_min) / 2

# ustaw zakresy osi symetrycznie
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

ax.set_box_aspect([1,1,1])
plt.show()
