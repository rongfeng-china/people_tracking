#!/usr/bin/env python

import cv2
import numpy as np
import time
import imutils
import datetime
from matplotlib import pyplot as plot

cap = cv2.VideoCapture("videos/Nov_19.mov")
while not cap.isOpened():
    cap = cv2.VideoCapture("videos/Nov_19.mov")
    cv2.waitKey(1000)
    print ("Wait for the header")

# The first frame
ret,img = cap.read()
newx, newy = img.shape[1]/2,img.shape[0]/2
img = cv2.resize(img,(newx,newy))
ret, thresh = cv2.threshold(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY) ,127, 255, cv2.THRESH_BINARY)
image, contours_pool, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
# mask for the pool
mask_pool = np.zeros((img.shape[0], img.shape[1],3),np.uint8)
for cnt_pool in contours_pool:
    cv2.drawContours(mask_pool, [cnt_pool], -1, (0,0,255), -1)


while(True):

    text = "Unoccupied"
    ret, img = cap.read()
    newx, newy = img.shape[1]/2, img.shape[0]/2
    img = cv2.resize(img,(newx,newy))
    if ret is False:
        break

    #img = cv2.pyrDown(img)
    ret, thresh = cv2.threshold(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY) , 127, 255, cv2.THRESH_BINARY)
    black = cv2.cvtColor(np.zeros((img.shape[1], img.shape[0]), dtype=np.uint8), cv2.COLOR_GRAY2BGR)
    # mask
    mask = np.zeros((img.shape[0],img.shape[1],3),np.uint8)
    # contour
    image, contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt_pool in contours_pool:
        cv2.drawContours(img, [cnt_pool], -1, (255,0 , 0), -1)
 
    for cnt in contours:
        epsilon = 0.01 * cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,epsilon,True)
        hull = cv2.convexHull(cnt)
        cv2.drawContours(img, [cnt], -1, (0, 255, 0), 2)
        cv2.drawContours(mask, [cnt], -1, (0, 0, 255), -1)
        #cv2.drawContours(img, [hull], -1, (0, 0, 255), 2)
    mask = mask - mask_pool
    # draw the text and timestamp on the frame
    cv2.putText(black, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
        (10, black.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        

    cv2.imshow("img", img)
    #cv2.imshow("black",black)
    cv2.imshow("mask",mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(0.015)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
