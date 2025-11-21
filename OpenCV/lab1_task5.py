import cv2
import numpy as np 
import os 

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

# this part makes it start up faster
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW, (cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY))
while not cap.isOpened():
    print("Can't open camera!!!! :(((")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)

blueMin = (80, 125, 25)
blueMax = (130, 255, 255)

cv2.namedWindow("Video Stream")

params = cv2.SimpleBlobDetector_Params()
params.thresholdStep = 255
params.minRepeatability = 1
params.blobColor = 255
params.filterByArea = False
params.filterByInertia = False
params.filterByConvexity = False
params.filterByCircularity = False

success, img = cap.read()
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, blueMin, blueMax)
detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(mask)
imagekp = cv2.drawKeypoints(mask, keypoints, None, color=(0,0,255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

while success and cv2.waitKey(1) == -1:
    cv2.imshow("Video Stream", imagekp)
    success, img = cap.read()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, blueMin, blueMax)
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(mask)
    imagekp = cv2.drawKeypoints(mask, keypoints, None, color=(0,0,255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    

cv2.destroyWindow("Video Stream")
cap.release()