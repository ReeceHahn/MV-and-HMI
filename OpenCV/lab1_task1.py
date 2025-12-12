import cv2
import numpy as np 

img = cv2.imread('gradient.png')

print(type(img))

print(img.shape)

cv2.namedWindow("Image")
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyWindow("Image")