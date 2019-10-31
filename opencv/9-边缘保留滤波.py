'''

美颜效果：边缘保留滤波 EPF
高斯双边
均值迁移
问我爱老红老公房间爱咖啡开始减肥设计费康师傅 福建省客服就开始尖峰时刻防守反击九分裤手机爱卡你好啊我是谁啊福啊是是方式蜀都赋我爱去o
'''


import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

def bi_demo(img):
    dst = cv.bilateralFilter(img,0,100,15)
    dst2 = cv.bilateralFilter(img,100,100,15)
    cv.imshow('dst',dst)
    cv.imshow('dst2',dst2)

def shift_demo(img):
    # 均值迁移
    dst = cv.pyrMeanShiftFiltering(img,10,50)
    cv.imshow('shift',dst)

img = cv.imread('lena.png')
cv.namedWindow('lena',cv.WINDOW_AUTOSIZE)
cv.imshow('lena',img)
bi_demo(img)
#shift_demo(img)
k = cv.waitKey(0)
if k == 27:
    cv.destroyAllWindows()