import cv2
import numpy as np
import csv

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
success = True

#Initialize lists for calibration points
objpoints = []
imgpoints = []

#Define the object point grid once (2.5 cm squares)
objp = np.zeros((9*6, 3), np.float32)
objp[:, :2] = 2.5 * np.mgrid[0:6, 0:9].T.reshape(-1, 2)

#Loop through the indices of images to be captured
for imgnum in range(number_of_images):

    #Capture images continuously and wait for a keypress
    while success and cv2.waitKey(1) == -1:

        #Read an image from the VideoCapture instance
        success, img = cap.read()

        #Convert image to greyscale for corner detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #Find chessboard corners
        ret, corners = cv2.findChessboardCorners(
            gray, (6, 9),
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        #Draw corners for viewing
        if ret == True:
            cv2.drawChessboardCorners(img, (6, 9), corners, ret)

        #Display the image
        cv2.imshow("Video Stream", img)

    #When we exit the capture loop we save the last image and repeat
    imglist.append(img)
    cv2.imshow("Captured Image", img)
    print("Image Captured")

    #Store object and image points only if corners were found
    if ret == True:
        subcorners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        objpoints.append(objp)
        imgpoints.append(subcorners)

#The image index loop ends when number_of_images have been captured
print("Captured", len(imglist), "images")

#Save all images to image files for later use
for imgnum, img in enumerate(imglist):
    cv2.imwrite("Image%03d.png" % (imgnum), img)

#Clean up the viewing window and release the VideoCapture instance
cv2.destroyWindow("Captured Image")
cv2.destroyWindow("Video Stream")
cap.release()

#Obtain the calibration parameters
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints,
    gray.shape[::-1], None, None
)

print("Intrinsic Matrix: \n")
print(mtx)
print("Distortion Coefficients: \n")
print(dist)
print("Rotation Vectors: \n")
print(rvecs)
print("Translation Vectors: \n")
print(tvecs)

#Save intrinsic matrix to CSV
with open('intrinsic_matrix.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerows(mtx)

#Save distortion coefficients to CSV
with open('distortion_coefficients.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerows(dist)