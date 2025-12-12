import cv2
import numpy as np

mtx = np.array([[763.30849663, 0, 628.40961392],
                [0, 762.61324942, 349.94376932],
                [0, 0, 1] ])

dist = np.array([[ 0.09051303, -0.2212114, -0.00074417, -0.0033674, 0.09970003]])

#Create a VideoCapture instance
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

#Check if the device could be opened and exit if not
if not cap.isOpened():
    print("Cannot open camera")
    exit()

#Set the resolution for image capture
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

#Set up windows for the video stream and captured images
cv2.namedWindow("Undistort")
cv2.namedWindow("Remapping")

while cv2.waitKey(1) == -1:
    #Read an image from the VideoCapture instance
    success, img = cap.read()

    # Undistort image using undistort function
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv2.imwrite("calibrated_undistortion.png", dst)

    # Undistort image using remapping
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
    dst2 = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    x, y, w, h = roi
    dst2 = dst2[y:y+h, x:x+w]
    cv2.imwrite("calibrated_remapping.png", dst2)

    #Display the images
    cv2.imshow("Undistort", dst)
    cv2.imshow("Remapping", dst2)

cv2.destroyWindow("Undistort")
cv2.destroyWindow("Remapping")
cap.release()