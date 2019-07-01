import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
img = cv.imread('lena.png')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.namedWindow('lena',cv.WINDOW_AUTOSIZE)
cv.imshow('lena',gray)

hrria=cv.cornerHarris(gray, 2, 3, 0.05)
#img1 = cv.threshold(hrria, 0.00001, 255, cv.THRESH_BINARY)
#print(img1)
#img[hrria>0.01*hrria.max()]=[0,0,255]
cv.imshow('harris', hrria)
k = cv.waitKey(0)
if k == 27:
    cv.destroyAllWindows()

plt.imshow(img, cmap='gray', interpolation='bicubic')