
'''
均值模糊
中值模糊
自定义模糊
'''
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

def mean_blur(img):
    # 卷积合（1，3）
    dst = cv.blur(img,(5,5))
    cv.imshow('dst',dst)

def median_blur(img):
    # 卷积合（1，3）
    dst = cv.medianBlur(img,5)
    cv.imshow('dst1',dst)
# 自定义模糊
def customer_blur(img):

    #kernel = np.ones((5,5),np.float32)/25
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], np.float32)
    dst = cv.filter2D(img, -1, kernel)
    cv.imshow('customer',dst)


img = cv.imread('lena.png')
cv.namedWindow('lena',cv.WINDOW_AUTOSIZE)
cv.imshow('lena',img)
mean_blur(img)
median_blur(img)
customer_blur(img)
k = cv.waitKey(0)
if k == 27:
    cv.destroyAllWindows()