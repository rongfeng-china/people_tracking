#!/usr/bin/env python
import cv2
import numpy as np
import time
import imutils
import datetime
from matplotlib import pyplot as plot

person_count = 0 
person_roi = np.zeros(shape=(10,4)) #maximum 10 persons
person_pre = np.zeros(shape=(10,4)) # prevous roi
person_flag = np.ones(shape=(10,1))
cap = cv2.VideoCapture("videos/Nov_19.mov")
while not cap.isOpened():
    cap = cv2.VideoCapture("videos/Nov_19.mov")
    cv2.waitKey(1000)
    print ("Wait for the header")

# The first frame
ret,img_pool = cap.read()
newx, newy = img_pool.shape[1]/2,img_pool.shape[0]/2
img_pool = cv2.resize(img_pool,(newx,newy))
ret, thresh_pool = cv2.threshold(cv2.cvtColor(img_pool.copy(), cv2.COLOR_BGR2GRAY) ,127, 255, cv2.THRESH_BINARY)
image, contours_pool, hier = cv2.findContours(thresh_pool, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
# mask for the pool
mask_pool = np.zeros((img_pool.shape[0], img_pool.shape[1],3),np.uint8)
for cnt_pool in contours_pool:
    cv2.drawContours(mask_pool, [cnt_pool], -1, (255,255,255), -1)

frame_num = 1
while(cap.isOpened() and frame_num < 3323):
    frame_num += 1
    text = "Unoccupied"
    ret, img = cap.read()
    newx, newy = img.shape[1]/2, img.shape[0]/2
    img = cv2.resize(img,(newx,newy))
    if ret is False:
        break

    ret, thresh = cv2.threshold(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY) , 127, 255, cv2.THRESH_BINARY)
   
    black = cv2.cvtColor(np.zeros((img.shape[1], img.shape[0]), dtype=np.uint8), cv2.COLOR_GRAY2BGR)
    # mask
    mask = np.zeros((img.shape[0],img.shape[1],3),np.uint8)
    # contour
    image, contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        if (len(cnt)>50):
		epsilon = 0.01 * cv2.arcLength(cnt,True)
        	approx = cv2.approxPolyDP(cnt,epsilon,True)
        	hull = cv2.convexHull(cnt)
                #cv2.drawContours(img, [cnt], -1, (0, 255, 0), 2)
        	cv2.drawContours(mask, [cnt], -1, (255,255,255), -1)
       		 #cv2.drawContours(img, [hull], -1, (0, 0, 255), 2)
    	mask = mask - mask_pool
    # mask filtering
    #kernel = np.ones((15,15),np.float32)/225
    #mask = cv2.filter2D(mask,-1,kernel)
    mask = cv2.medianBlur(mask,7)
    # human detection and boundingbox
    ret_hm, thresh_hm = cv2.threshold(cv2.cvtColor(mask.copy(), cv2.COLOR_BGR2GRAY) ,127, 255, cv2.THRESH_BINARY)
        
    image_hm, contours_hm, hier_hm = cv2.findContours(thresh_hm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    # human detection
    count = 0
    for cnt_hm in contours_hm:
        count += 1
	x,y,w,h = cv2.boundingRect(cnt_hm)
	if(person_count == 0):
            print ("first person")
            person_count+=1
            person_roi[person_count] = [x,y,w,h]
            person_flag[person_count] = 1
        else:
            previous_flag = 0 # current person appears before? yes/no 1/0
            for num in range(1,person_count+1):
                if( person_flag[num] != 0):
                    x_val = person_roi[num][0]+person_roi[num][2]/2
                    y_val = person_roi[num][1]+person_roi[num][3]/2
                    if(person_count == 2):
                        print( "2 person")
                        print num,x+w/2, x_val
                    if(abs(x+w/2-x_val) < 50 and abs(y+h/2-y_val) < 100):
		        #update the personindex with num
			if (person_count == 2):
		            print("update")
                            print (num)
			person_pre[num] = person_roi[num]
		        person_roi[num] = [x,y,w,h]
                        person_flag[num] = 1 # you can delete this line
                        # previous person leaving
		        #print x,newx-50,person_pre[num][0]
		        if((x+w < 20 and person_pre[num][0] > x)  or (x > newx-20 and person_pre[num][0] < x) or (y+h < 15 and person_pre[num][1] > y) or( y > newy-15 and person_pre[num][1] < y)):
                            # leaving
                            print ("leaving")
			    person_count -= 1
                            for i in range(num,person_count+1):
                                person_roi[i]=person_roi[i+1]
                                person_flag[i] = person_flag[i+1]
                        previous_flag = 1
            if (previous_flag == 0):
                if(x+w < 20 or x > newx-20 or y+h < 15 or y > newy -15 ):
                    # new coming person
		    print("new person coming")
		    person_count += 1
                    print (person_count)
                    person_roi[person_count] = [x,y,w,h]
                    person_flag[person_count] = 1
                     
	#print ("human:%d, center" %(count)),
        #print (x+w/2,y+h/2),
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),1)
    #if (count != 0):
    #    print('\n')    
    # draw blue pool
    for cnt_pool in contours_pool:
        cv2.drawContours(img, [cnt_pool], -1, (255,0 , 0), -1)
   
    # text for counting
    cv2.putText(img,"Frame%d" %(frame_num),(10,10),cv2.FONT_HERSHEY_SIMPLEX,0.35,(0,0,255),1)
    cv2.putText(img,"People In Area:%d" %(person_count), (10, img.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    cv2.putText(img,"In Pool:0", (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    

    cv2.imshow("img", img)
    #cv2.imshow("thresh",img_new)
    cv2.imshow("mask",mask)
    cv2.waitKey(100)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(0.015)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
