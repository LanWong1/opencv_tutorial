import cv2 as cv
import numpy as np

img1 = cv.imread('lena.png')
img2 = cv.imread('2.jpg')

rows, cols, channels = img1.shape
roi = img2[0:rows, 0:cols]

img1gray = cv.cvtColor(img1,cv.COLOR_RGB2GRAY)
ret, mask = cv.threshold(img1gray, 100, 255, cv.THRESH_BINARY)

mask_inv = cv.bitwise_not(mask)

img2_bg = cv.bitwise_and(roi, roi, mask = mask)
img1_fg = cv.bitwise_and(img1, img1, mask = mask_inv)


dst = cv.add(img2_bg, img1_fg)
img2[0:rows, 0:cols] = dst
cv.imshow('img1gray',img1gray)
cv.imshow('mask', mask)
# cv.imshow('mask_inv',mask_inv)
# cv.imshow('img_bg', img2_bg)
# cv.imshow('im_fg',img1_fg)
# cv.imshow('dst', dst)
# cv.imshow('result',img2)

cv.waitKey(0)
cv.destroyAllWindows()