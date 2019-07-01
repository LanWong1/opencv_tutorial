import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np


def fill_color_demo(img):
    copyImg = img.copy()
    h, w = img.shape[:2]
    mask = np.zeros([h+2, w+2],np.uint8)
    #
    cv.floodFill(copyImg,mask,(200,200),(0,255,255),(100,100,100),(50,50,50),cv.FLOODFILL_FIXED_RANGE)
    print(mask)
    cv.imshow('filled_image',copyImg)

def fill_binary_demo(img):
    # img = np.zeros([400,400,3],np.uint8)
    # img[100:300, 100:300, :] = 255
    h,w = img.shape[:2]
    print(h,w)
    mask = np.ones([h+2, w+2],np.uint8)
    mask[101:301, 101:301] = 0
    cv.floodFill(img,mask,(100,100),(0,255,0),cv.FLOODFILL_MASK_ONLY)
    cv.imshow('image',img)


img = cv.imread('lena.png')
cv.namedWindow('lena',cv.WINDOW_AUTOSIZE)
cv.imshow('lena',img)
# face = img[100:250, 100:300]
# gray = cv.cvtColor(face, cv.COLOR_BGR2GRAY)
#
# backface = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
# img[100:250, 100:300] = backface
# cv.imshow('face1',img)

#fill_color_demo(img)
fill_binary_demo(img)
k = cv.waitKey(0)
if k == 27:
    cv.destroyAllWindows()
