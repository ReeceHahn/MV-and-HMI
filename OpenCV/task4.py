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

blueMin = (80, 130, 30)
blueMax = (130, 240, 255)

cv2.namedWindow("Video Stream")
success, img = cap.read()
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, blueMin, blueMax)
masked_image = cv2.bitwise_and(img, img, mask=mask)
inverted_mask = cv2.bitwise_not(mask)

while success and cv2.waitKey(1) == -1:
    cv2.imshow("Video Stream", masked_image)
    success, img = cap.read()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, blueMin, blueMax)
    masked_image = cv2.bitwise_and(img, img, mask=mask)
    inverted_mask = cv2.bitwise_not(mask)

cv2.destroyWindow("Video Stream")
cap.release()