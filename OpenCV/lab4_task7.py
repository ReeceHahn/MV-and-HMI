import cv2
import numpy as np
import matplotlib.pyplot 

mtx = np.array([[763.30849663, 0, 628.40961392],
                [0, 762.61324942, 349.94376932],
                [0, 0, 1] ])

dist = np.array([[ 0.09051303, -0.2212114, -0.00074417, -0.0033674, 0.09970003]])

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
number_of_images = 2
imglist = []
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
    cv2.imshow("Captured Image", img)
    print("Image Captured")

    h, w = img.shape[:2]
    newcameramtx, (x, y, w, h) = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    img = dst[y:y+h, x:x+w]
    imglist.append(img)
    
#The image index loop ends when number_of_images have been captured
print("Captured", len(imglist), "images")

if(len(imglist) > 1):
    gray1 = cv2.cvtColor(imglist[-1], cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(imglist[-2], cv2.COLOR_BGR2GRAY)
    stereo = cv2.StereoBM.create(numDisparities = 16, blockSize = 21)
    disparity = stereo.compute(gray1, gray2)
    matplotlib.pyplot.imshow(disparity, 'gray')
    matplotlib.pyplot.show()

#Save all images to image files for later use
for imgnum, img in enumerate(imglist):
    cv2.imwrite("Image%03d.png" % (imgnum), img)

#Clean up the viewing window and release the VideoCapture instance
cv2.destroyWindow("Captured Image")
cv2.destroyWindow("Video Stream")

cap.release()