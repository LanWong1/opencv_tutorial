import cv2
import numpy as np
# 灰度化读取图片
image_1 = cv2.imread('12.jpg')
#cv2.cvtColor(image_1,cv2.COLOR_BGR2GRAY)
gray = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
height, width = image_1.shape


# 将图片二值化
retval, img = cv2.threshold(image_1,127,255,cv2.THRESH_BINARY)
cv2.imshow('image',img)
# 创建一个空白图片(img.shape[0]为height,img.shape[1]为width)
paintx = np.zeros(img.shape,np.uint8)




# 创建width长度都为0的数组
w = [0]*image.shape[1]


# 对每一行计算投影值
for x in range(width):
    for y in range(height):
# t = cv2.cv.Get2D(cv2.cv.fromarray(img),y,x)
        k = img.item(y,x)
        if k == 0:
            w[x] += 1

# 绘制垂直投影图
print(w)
for x in range(height):
    for i in range(len(w)):
# 把大于0的像素变成白
        if w[i] > 10:
            paintx[x,i] = 255
cv2.imshow('paintx',paintx)
cv2.waitKey(0)
