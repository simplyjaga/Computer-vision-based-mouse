import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pyautogui 

cam = cv.VideoCapture(0)

lower_yellow = np.array([10,50,100])
upper_yellow = np.array([30,180,255])

lower_purple= np.array([110,60,30])
upper_purple = np.array([150,130,100])

while (True):
    # frame size is 480 x 640

    bool, frame = cam.read()
    frame = cv.flip(frame, 1)
    frame = cv.GaussianBlur(frame,(7,7),0)

    mask = np.zeros_like(frame)
    mask[100:400,100:400] =[255,255,255]


    #finding roi
    img_roi = cv.bitwise_and(frame,mask)

    #drawing grid in frame
    #box
    cv.rectangle(frame, (100,100),(400,400),(0,255,0),3)
    #vertical lines
    cv.line(frame,(200,100),(200,400),(0,255,0),3)
    cv.line(frame,(300,100),(300,400),(0,255,0),3)
    #horizontal lines
    cv.line(frame,(100,200),(400,200),(0,255,0),3)
    cv.line(frame,(100,300),(400,300),(0,255,0),3)

    # finding contours for roi
    img_roi_hsv = cv.cvtColor(img_roi,cv.COLOR_BGR2HSV)
    
    # mouse pointer 
    mouse_pointer_thres = cv.inRange(img_roi_hsv,lower_yellow,upper_yellow)
    contours,hierarchy = cv.findContours(mouse_pointer_thres,cv.RETR_TREE,
                            cv.CHAIN_APPROX_SIMPLE)
    if (len(contours)!= 0):
        areas = [cv.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt = contours[max_index]
        # getting centroid of our contour
        M = cv.moments(cnt)
        # for ignoring noise use area condition in if statement
        if (M['m00'] != 0  and areas[max_index] > 500):
            centroid_x = int(M['m10']/M['m00'])
            centroid_y = int(M['m01']/M['m00'])
            # drawing markerpoint in frame
            cv.circle(frame, (centroid_x,centroid_y), 5,(255,0,0),-1)
            #cursor actions
            if centroid_x < 200 :
                dist_x = -5
            elif centroid_x > 300 : 
                dist_x = 5
            else:
                dist_x = 0

            if centroid_y < 200 :
                dist_y = -5
            elif centroid_y > 300 : 
                dist_y = 5
            else:
                dist_y = 0

            pyautogui.moveRel(dist_x,dist_y,duration=0.10)

    
    # mouse button 
    mouse_button_thres = cv.inRange(img_roi_hsv,lower_purple,upper_purple)
    contours,hierarchy = cv.findContours(mouse_button_thres,cv.RETR_TREE,
                            cv.CHAIN_APPROX_SIMPLE)
    if (len(contours)!= 0):
        areas = [cv.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt = contours[max_index]
        # getting centroid of our contour
        M = cv.moments(cnt)
        # for ignoring noise use area condition in if statement
        if (M['m00'] != 0  and areas[max_index] > 2000):
            centroid_x = int(M['m10']/M['m00'])
            centroid_y = int(M['m01']/M['m00'])
            # drawing markerpoint in frame
            cv.circle(frame, (centroid_x,centroid_y), 5,(0,0,255),-1)
            pyautogui.click()
            cv.waitKey(10)


    cv.imshow("frame",frame)
    # cv.imshow("mouse_pointer",mouse_pointer_thres)
    # cv.imshow("mouse_button",mouse_button_thres)

    key = cv.waitKey(10)
    if key == ord('q'):
        break

cam.release()
cv.destroyAllWindows()