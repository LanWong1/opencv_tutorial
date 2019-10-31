import cv2 as cv
#from matplotlib import pyplot as plt
import numpy as np



img = cv.imread('lena.png')
img = np.array(img).astype(np.float32) / 255.0 - 0.5
print(img)
cv.namedWindow('lena',cv.WINDOW_AUTOSIZE)
cv.imshow('lena',img)
k = cv.waitKey(0)
if k == 27:
    cv.destroyAllWindows()
