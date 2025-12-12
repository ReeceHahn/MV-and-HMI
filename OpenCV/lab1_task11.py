import cv2
import numpy as np 

img = cv2.imread('Bismarck.jpg', cv2.IMREAD_GRAYSCALE)

# low pass 5x5 guassian 
low_pass_img = cv2.GaussianBlur(img, (5, 5), 0)

fast = cv2.FastFeatureDetector_create()

fast.setNonmaxSuppression(0)

kp = fast.detect(low_pass_img, None)

orb = cv2.ORB_create()

kp = orb.detect(low_pass_img, None)

kp, des = orb.compute(low_pass_img, kp)

orb_low_pass = cv2.drawKeypoints(low_pass_img, kp, None, color=(0,255,0), flags=0)

## high pass canny
high_pass_img = cv2.Canny(img, 40, 235)

fast = cv2.FastFeatureDetector_create()

fast.setNonmaxSuppression(0)

kp = fast.detect(high_pass_img, None)

orb = cv2.ORB_create()

kp = orb.detect(high_pass_img, None)

kp, des = orb.compute(high_pass_img, kp)

orb_high_pass = cv2.drawKeypoints(high_pass_img, kp, None, color=(0,255,0), flags=0)

# threshold
thresh_value,thresh_img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

fast = cv2.FastFeatureDetector_create()

fast.setNonmaxSuppression(0)

kp = fast.detect(thresh_img, None)

orb = cv2.ORB_create()

kp = orb.detect(thresh_img, None)

kp, des = orb.compute(thresh_img, kp)

orb_thresh = cv2.drawKeypoints(thresh_img, kp, None, color=(0,255,0), flags=0)

cv2.namedWindow("Image")

while True:
    cv2.imshow("Image", img)
    if cv2.waitKey(0) == ord('q'):
        break
    cv2.imshow("Image", orb_low_pass)
    if cv2.waitKey(0) == ord('q'):
        break
    cv2.imshow("Image", orb_high_pass)
    if cv2.waitKey(0) == ord('q'):
        break
    cv2.imshow("Image", orb_thresh)
    if cv2.waitKey(0) == ord('q'):
        break

cv2.destroyWindow("Image")