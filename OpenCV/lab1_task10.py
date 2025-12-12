import cv2
import numpy as np 

img = cv2.imread('Bismarck.jpg', cv2.IMREAD_GRAYSCALE)

fast = cv2.FastFeatureDetector_create()

fast.setNonmaxSuppression(0)

kp = fast.detect(img, None)

kp_img = cv2.drawKeypoints(img, kp, None, color=(255, 0, 0))

orb = cv2.ORB_create()

kp = orb.detect(img, None)

kp, des = orb.compute(img, kp)

orb_img = cv2.drawKeypoints(kp_img, kp, None, color=(0,255,0), flags=0)

cv2.namedWindow("Image")

while True:
    cv2.imshow("Image", img)
    if cv2.waitKey(0) == ord('q'):
        break
    cv2.imshow("Image", kp_img)
    if cv2.waitKey(0) == ord('q'):
        break
    cv2.imshow("Image", orb_img)
    if cv2.waitKey(0) == ord('q'):
        break

cv2.destroyWindow("Image")