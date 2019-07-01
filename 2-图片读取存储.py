import cv2 as cv
import numpy as np


def video_demo():
    capture = cv.VideoCapture(0)
    while(True):

        ret, frame = capture.read()
        frame = cv.flip(frame, 1)
        cv.imshow('video',frame)
        c = cv.waitKey(10)
        if c == 30:
            break

img = cv.imread('lena.png')
cv.namedWindow('lena',cv.WINDOW_AUTOSIZE)
cv.imshow('lena',img)
cv.waitKey(0)
cv.destroyAllWindows()
cv.imwrite('result.png',img) #保存图片
#video_demo()