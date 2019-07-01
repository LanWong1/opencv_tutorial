import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

def gradiant_demo(img):
    # grad_x = cv.Sobel(img, cv.CV_32F, 1, 0)
    # grad_y = cv.Sobel(img, cv.CV_32F, 0, 1)
    grad_x = cv.Scharr(img, cv.CV_32F, 1, 0)
    grad_y = cv.Scharr(img, cv.CV_32F, 0, 1)
    grad_x = cv.convertScaleAbs(grad_x)
    grad_y = cv.convertScaleAbs(grad_y)
    cv.imshow('gradx',grad_x)
    cv.imshow('grady',grad_y)
    grad_xy = cv.addWeighted(grad_x, 0.5, grad_y,0.5,0)
    cv.imshow('grad_xy',grad_xy)






img = cv.imread('lena.png')
cv.namedWindow('lena',cv.WINDOW_AUTOSIZE)
cv.imshow('lena',img)
gradiant_demo(img)
k = cv.waitKey(0)
if k == 27:
    cv.destroyAllWindows()

plt.imshow(img, cmap='gray', interpolation='bicubic')