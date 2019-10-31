'''
滤镜做高斯模糊
'''


import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
img = cv.imread('lena.png')
img1 = cv.GaussianBlur(img,(3,3),0)
cv.namedWindow('lena',cv.WINDOW_AUTOSIZE)
cv.imshow('lena',img)
cv.imshow('gausiin',img1)
k = cv.waitKey(0)