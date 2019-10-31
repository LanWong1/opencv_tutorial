import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np



# img = cv.imread('lena.png')
img = np.zeros((512,512,3),np.uint8)

# cv.namedWindow('lena',cv.WINDOW_AUTOSIZE)
# cv.imshow('lena',img)

cv.line(img,(0,0),(511,511),(255,0,0),5)
# 左上右下顶点 颜色为绿色  线宽3
cv.rectangle(img,(384,0),(510,128),(0,255,0),3)
# 画圆 圆心坐标 半径大小 颜色 线宽
cv.circle(img,(447,63),63,(0,0,255),-1)
# 椭圆 中心坐标 长轴短轴,角度
cv.ellipse(img,(256,256),(100,50), 0, 0, 180, 255, -1)
# 多边形需要指定每个顶点的坐标
pst  = np.array([[10,5],[20,30],[70,20],[50,10]],np.int32)
pst  = pst.reshape((-1,1,2))
cv.polylines(img, [pst], True, (0,255,255))
# 添加文字 内容 位置,字体,字体大小 颜色
font = cv.FONT_HERSHEY_COMPLEX
cv.putText(img,'I love Lan',(10,500),font,4,(0,255,255),lineType=cv.LINE_AA)
cv.imshow('img',img)

k = cv.waitKey(0)
if k == 27:
    cv.destroyAllWindows()