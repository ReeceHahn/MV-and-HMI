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

cv2.namedWindow("Video Stream")
success, img = cap.read()

while success and cv2.waitKey(1) == -1:
    cv2.imshow("Video Stream", img)
    success, img = cap.read()

cv2.destroyWindow("Video Stream")
cap.release()