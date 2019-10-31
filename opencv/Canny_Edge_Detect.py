'''
canny边缘检测:
1.转为灰度图
2.高斯滤波
3.求梯度
4.非极大值抑制
5.选择最大最小值
'''
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt



img = cv.imread('lena.png', 0)
#cv.namedWindow('lena',cv.WINDOW_AUTOSIZE)
cv.imshow('lena',img)
edges = cv.Canny(img,100,200)
cv.imshow('edges',edges)
k = cv.waitKey(0)
if k == 27:
    cv.destroyAllWindows()
