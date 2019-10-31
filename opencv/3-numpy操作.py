'''
图像即位np的数组
'''

import cv2 as cv
import numpy as np


def create_demo():
    img = np.ones([400,400],np.uint8)
    img = img * 127 # 灰色图
    cv.imshow('demo', img)

# img = cv.imread('lena.png')
# cv.namedWindow('lena',cv.WINDOW_AUTOSIZE)
# cv.imshow('lena',img)
# img2 = 255 - img
# cv.imshow('img2',img2)
create_demo()
cv.waitKey(0)
cv.destroyAllWindows()

