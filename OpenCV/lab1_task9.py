import cv2
import numpy as np 

img = cv2.imread('Bismarck.jpg', cv2.IMREAD_GRAYSCALE)

fast = cv2.FastFeatureDetector_create()

fast.setNonmaxSuppression(0)

kp = fast.detect(img, None)

kp_img = cv2.drawKeypoints(img, kp, None, color=(255, 0, 0))

print(len(kp))
print(kp[10].pt)

cv2.namedWindow("Image")

while True:
    cv2.imshow("Image", img)
    if cv2.waitKey(0) == ord('q'):
        break
    cv2.imshow("Image", kp_img)
    if cv2.waitKey(0) == ord('q'):
        break     

cv2.destroyWindow("Image")