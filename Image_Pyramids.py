'''
1.高斯
2.删除偶数行和列
'''

import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np


img = cv.imread('lena.png')
cv.namedWindow('lena',cv.WINDOW_AUTOSIZE)


cv.imshow('lena',img)
lower_reso = cv.pyrDown(img)
higher_reso = cv.pyrUp(img)
low2 = cv.pyrDown(lower_reso)
cv.imshow('low2',low2)
cv.imshow('low',lower_reso)
cv.imshow('high',higher_reso)


k = cv.waitKey(0)
if k == 27:
    cv.destroyAllWindows()