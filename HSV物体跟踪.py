
# coding: utf-8

# In[ ]:


import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FOURCC,cv.VideoWriter_fourcc('M','J','P','G'))
cap.set(cv.CAP_PROP_FRAME_WIDTH,1920)
cap.set(cv.CAP_PROP_FRAME_HEIGHT,1080)



while(1):
    ret, frame = cap.read()
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # 设定阈值
    lower_blue = np.array([110, 50, 50])
    higher_blue = np.array([130,255,255])
    mask = cv.inRange(hsv,lower_blue, higher_blue)

    res = cv.bitwise_and(frame, frame, mask=mask)
    cv.imshow('frame', frame)
    cv.imshow('mask',mask)
    cv.imshow('res', res)
    k = cv.waitKey(5)&0xFF
    if k==27:
        break
cv.destroyAllWindows()



    

