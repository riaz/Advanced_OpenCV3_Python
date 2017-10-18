import numpy as np
import cv2
import os
import glob
import datetime as datetime


PATTERN_PATH = os.path.join("/","opencv","samples","data")

criteria = (cv2.TERM_CRITERIA_EPS  + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# preparing object points with Z = 0
objp = np.zeros((6*7,3), np.float32)

#print(objp.shape)

objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

#print(objp)
#print(objp.shape)

obj_points = []
image_points = []

for image in glob.glob(os.path.join(PATTERN_PATH,"left*.jpg")):
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (7,6), None)

    if ret == True:
        obj_points.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        corners2 = np.reshape(corners2, (42,2))
        print(np.array(corners2).shape)
        image_points.append(corners2)

        # Drawing and displaying the corners
        img = cv2.drawChessboardCorners(img, (7,6), corners2, ret)
        #cv2.imwrite('image.jpg',img)

# At this stage we have both the object points and the image points
# next, we will need to calibrate the camera using the image and the object points

img = cv2.imread(os.path.join(PATTERN_PATH,"left12.jpg"))
h, w = img.shape[:2]

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, image_points, (h,w), None, None)

# generating a new camera matrix based a free scaling parameter, alpha
# we will use the function getOptiomalNewCameraMatrix()

new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h),  1, (w,h))

# undistorting the image
dst = cv2.undistort(img, mtx, dist, None, new_camera_mtx)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite("corrected.jpg",dst)
cv2.imwrite("normal.jpg",img) 



