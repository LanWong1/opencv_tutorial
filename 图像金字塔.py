import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np


def primid_demo(img):
    primid_imags=[]
    level = 3
    source  = img
    for i in range(level):
        dst = cv.pyrDown(source)
        cv.imshow('primid'+str(i), dst)
        primid_imags.append(dst)
        source = dst
    return primid_imags



def laplas_demo(img):
    primid_img = primid_demo(img)
    for i in range(len(primid_img)-1, -1, -1):
        # expand = []
        laplas = []
        if i == 0:
            expand = cv.pyrUp(primid_img[i],dstsize=img.shape[:2])
            laplas  = cv.subtract(img,expand)
        else:
            expand = cv.pyrUp(primid_img[i], dstsize=primid_img[i - 1].shape[:2])
            laplas = cv.subtract(primid_img[i - 1], expand)
        cv.imshow('laplas' + str(i), laplas)


img = cv.imread('lena.png')
print(img.shape)
cv.namedWindow('lena',cv.WINDOW_AUTOSIZE)
cv.imshow('lena',img)

#primid_demo(img)
laplas_demo(img)



k = cv.waitKey(0)
if k == 27:
    cv.destroyAllWindows()

plt.imshow(img, cmap='gray', interpolation='bicubic')