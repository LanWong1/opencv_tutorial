import cv2
import numpy as np
from matplotlib import pyplot as plt
import copy
img = cv2.imread('sfz1.jpeg',0)
#img2 = img.copy()
templete = cv2.imread('ddd.jpg',0)
w, h = templete.shape[::-1]
res = cv2.matchTemplate(img, templete,cv2.TM_SQDIFF)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(img,top_left, bottom_right, 255, 2)

cv2.imshow('res',img)
cv2.waitKey(0) 