'''
色彩空间 RGB 255，255，255是白色
HSV    H：0-180 S：0-255 V：0-255
HIS
YCrCb 提取人的皮肤
YUV  linux默认色彩空间
最常见 HSV和BGR YUV与BGR
'''
import cv2 as cv
import numpy as np

def extract_obj_demo():

    capture = cv.VideoCapture(0)
    while(True):
        ret, frame = capture.read()
        if ret == False:
            break
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        lower_hsv = np.array([0,0,46])
        high_hsv = np.array([180,43,220]) #灰色
        mask =  cv.inRange(hsv,lowerb= lower_hsv,upperb= high_hsv)# 二值图像
        dst = cv.bitwise_and(frame,frame, mask=mask)#显示灰色

        # print(frame, mask)
        cv.imshow('vedio ',frame)
        cv.imshow('video1',dst)
        c = cv.waitKey(10)
        if c == 27:
            capture.release()
            cv.destroyAllWindows()
            break




def color_space_demo(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow('gray',gray)
    HSV  = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    cv.imshow('HSV',HSV)
    YUV = cv.cvtColor(img,cv.COLOR_BGR2YUV)
    cv.imshow('YUV',YUV)




# img = cv.imread('lena.png')
# cv.namedWindow('lena',cv.WINDOW_AUTOSIZE)
# cv.imshow('lena',img)
# # color_space_demo(img)
# b,g,r = cv.split(img)
# cv.imshow('b',b)
# cv.imshow('g',g)
# cv.imshow('r',r)
# src = cv.merge([b,g,r])
# src[:,:,0] = 0
# cv.imshow('src',src)
# cv.waitKey(0)
# cv.destroyAllWindows()
extract_obj_demo()
