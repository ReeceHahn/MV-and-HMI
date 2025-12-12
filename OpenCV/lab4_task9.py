import cv2
import numpy as np
import matplotlib.pyplot 

mtx = np.array([[762.18176135, 0, 630.79869476],
                [0, 761.58191045, 352.74859373],
                [0, 0, 1] ])

dist = np.array([[ 9.40983824e-02, -2.28597400e-01, -2.14388468e-04, -1.45343212e-03, 6.59170231e-02]])

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
imgpoints = []
imgdescs = [] 
success = True

#Loop through the indices of images to be captured
for imgnum in range(number_of_images):
    
    #Capture images continuously and wait for a keypress
    while success and cv2.waitKey(1) == -1:
        
        #Read an image from the VideoCapture instance
        success, img = cap.read()

        orb = cv2.ORB_create()
        kp = orb.detect(img, None)
        kp, des = orb.compute(img, kp)
        img_orb = cv2.drawKeypoints(img, kp, None, flags=0)
        
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
    imgpoints.append(kp)
    imgdescs.append(des)

    if(len(imglist) > 1):

        img1 = imglist[-1]
        img2 = imglist[-2]
        kp1 = imgpoints[-1]
        kp2 = imgpoints[-2]
        des1 = imgdescs[-1]
        des2 = imgdescs[-2]

        gray1 = cv2.cvtColor(imglist[-1], cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(imglist[-2], cv2.COLOR_BGR2GRAY)
        stereo = cv2.StereoBM.create(numDisparities = 16, blockSize = 21)
        disparity = stereo.compute(gray1, gray2)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key = lambda x:x.distance)
        img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        pts1 = []
        pts2 = []

        for m in matches:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

            pts2 = np.int32(pts2)
            pts1 = np.int32(pts1)

        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)

        print(F)

        E, mask = cv2.findEssentialMat(pts1, pts2, mtx, cv2.LMEDS, prob=0.999)

        print(E)

        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, mtx)

        print(R)
        print(t)

        #matplotlib.pyplot.imshow(disparity, 'gray')
        #matplotlib.pyplot.show()

        #while cv2.waitKey(1) == -1:
            #cv2.imshow("Video Stream", img_matches)

#The image index loop ends when number_of_images have been captured
print("Captured", len(imglist), "images")

#Save all images to image files for later use
for imgnum, img in enumerate(imglist):
    cv2.imwrite("Image%03d.png" % (imgnum), img)

#Clean up the viewing window and release the VideoCapture instance
cv2.destroyWindow("Captured Image")
cv2.destroyWindow("Video Stream")

cap.release()