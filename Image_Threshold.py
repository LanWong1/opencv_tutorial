
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

def threshold_demo(img):
    ret, thresh1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)# >127,255, or 0
    ret, thresh2 = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV) # >127,0, or 255
    ret, thresh3 = cv.threshold(img, 127, 255, cv.THRESH_TRUNC) # >127 , =127, or 不变
    ret, thresh4 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO) # >127 , 不变, or 0
    ret, thresh5 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO_INV) # >127, =0, or 不变

    windownames = ['BINARAY', 'BINARAY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
    imgs = [thresh1,thresh2,thresh3, thresh4, thresh5]
    for i in range(5):
        cv.imshow(windownames[i],imgs[i])

def adaptive_threshold_demo(img):
    img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret, thresh1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    cv.imshow('THRESH_BINARY', thresh1)
    # 适用于灰度图 cv.ADAPTIVE_THRESH_MEAN_C threshold = 区域内均值
    thresh2 = cv.adaptiveThreshold(img, 205, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 5, 2)
    cv.imshow('THRESH_MEAN_C', thresh2)
    # ADAPTIVE_THRESH_GAUSSIAN_C threshold = 高斯窗口的权重的加权和
    thresh3 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2) # block size = 11
    cv.imshow('THRESH_GAUSSIAN_C', thresh3)






img = cv.imread('lena.png')
cv.namedWindow('lena',cv.WINDOW_AUTOSIZE)
cv.imshow('lena',img)
#threshold_demo(img)
adaptive_threshold_demo(img)
k = cv.waitKey(0)
if k == 27:
    cv.destroyAllWindows()