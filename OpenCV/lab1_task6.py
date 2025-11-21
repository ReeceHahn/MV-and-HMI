import cv2
import numpy as np 

img = cv2.imread('Bismarck.jpg', cv2.IMREAD_GRAYSCALE)

x = 5
y = 5

blur_img = cv2.GaussianBlur(img, (x, y), 0)

box_filter = cv2.boxFilter(img, -1, (x ,y))

cv2.namedWindow("Image")

while True:
    cv2.imshow("Image", img)
    if cv2.waitKey(0) == ord('q'):
        break
    cv2.imshow("Image", blur_img)
    if cv2.waitKey(0) == ord('q'):
        break
    cv2.imshow("Image", box_filter)
    if cv2.waitKey(0) == ord('q'):
        break

cv2.destroyWindow("Image")