import cv2
import numpy as np

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
cv2.namedWindow("Video Stream")
cv2.namedWindow("Captured Image")

#Initialize an empty list of images and the number to be captured
number_of_images = 10
imglist = []
objpoints = []
imgpoints = []
success = True

#Loop through the indices of images to be captured
for imgnum in range(number_of_images):
    
    #Capture images continuously and wait for a keypress
    while success and cv2.waitKey(1) == -1:
        
        #Read an image from the VideoCapture instance
        success, img = cap.read()
        
        #Display the image
        cv2.imshow("Video Stream", img)
        
    #When we exit the capture loop we save the last image and repeat
    imglist.append(img)
    cv2.imshow("Captured Image", img)
    print("Image Captured")
    
#The image index loop ends when number_of_images have been captured
print("Captured", len(imglist), "images")

#Save all images to image files for later use
for imgnum, img in enumerate(imglist):
    cv2.imwrite("Image%03d.png" % (imgnum), img)

#Clean up the viewing window and release the VideoCapture instance
cv2.destroyWindow("Captured Image")
cv2.destroyWindow("Video Stream")

for img in imglist:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (6,9), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    objp = np.zeros((9*6,3), np.float32)
    objp[:,:2] = 2.5*np.mgrid[0:6,0:9].T.reshape(-1,2)

    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

    subcorners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

    cv2.drawChessboardCorners(img, (6,9), corners, ret)
    cv2.imshow("Points", img)

    while cv2.waitKey(1) == -1: ()

cv2.destroyWindow("Points")

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("Intrinsic Matrix: \n")
print(mtx)
print("Distortion Coefficients: \n")
print(dist)
print("Rotation Vectors: \n")
print(rvecs)
print("Translation Vectors: \n")
print(tvecs)

cap.release()