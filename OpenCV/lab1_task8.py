import cv2
import numpy as np 

img = cv2.imread('Bismarck.jpg', cv2.IMREAD_GRAYSCALE)

edges = cv2.Canny(img,30,265)

cv2.namedWindow("Image")

while True:
    cv2.imshow("Image", img)
    if cv2.waitKey(0) == ord('q'):
        break
    cv2.imshow("Image", edges)
    if cv2.waitKey(0) == ord('q'):
        break

cv2.destroyWindow("Image")