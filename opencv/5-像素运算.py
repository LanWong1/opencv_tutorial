'''
算术运算  ： 加减乘除，调节亮度，对比度
逻辑运算  ：与或非 遮罩层控制
'''
import cv2 as cv
import numpy as np

def caculate(m1,m2):
    add = cv.add(m1,m2)
    cv.imshow('add',add)

    min = cv.subtract(m1,m2)
    cv.imshow('subs',min)

    devide = cv.divide(m1,m2)
    cv.imshow('devide',devide)

    mult = cv.multiply(m1, m2)
    cv.imshow('mult',mult)

def other_caculate(m1,m2):
    aver1 = cv.mean(m1) # 均值可以找到主色彩 B，G，R对应三个均值，均值越大，颜色越突出
    aver2 = cv.mean(m2)
    M1,dev1 = cv.meanStdDev(m1)#方差越大，颜色变化越剧烈
    M2,dev2 = cv.meanStdDev(m2)
    print(M1,dev1,M2,dev2)



def logic_caculate(m1, m2):
    dst = cv.bitwise_and(m1,m2)
    cv.imshow('dst',dst)

    dst1 = cv.bitwise_not(m1,m2)
    cv.imshow('dst1',dst1)

def contract_brightness_demo(img, c, b):
    h, w, ch = img.shape
    blank  = np.zeros([h, w, ch], img.dtype)
    dst = cv.addWeighted(img, c, blank, 1-c, b)
    # dst = cv.addWeighted(img,c)
    cv.imshow('dst', dst)

img1 = cv.imread('1.jpg')
img2 = cv.imread('2.jpg')
cv.namedWindow('img1',cv.WINDOW_AUTOSIZE)
cv.imshow('img1',img1)
cv.imshow('img2',img2)
contract_brightness_demo(img2,1.7,0)
# caculate(img1,img2)
# logic_caculate(img1,img2)
# other_caculate(img1,img2)
cv.waitKey(0)
cv.destroyAllWindows()