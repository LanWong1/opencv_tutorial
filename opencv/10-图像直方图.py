'''
直方图 histogram
均衡化 图像增强的手段 对比度增强
比较
反向投影
'''




import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np


def plot_demo(img):
    plt.hist(img.ravel(),256,[0,256])
    plt.show()

def histogram_demo(img):
    colors = ('blue','green','red')
    for i, color in enumerate(colors):
        hist = cv.calcHist([img],[i],None,[256],[0,256])
        plt.plot(hist,color=color)
        plt.xlim([0,256])
    plt.show()
# 全局直方图均衡化
def equalHis_demo(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    dst = cv.equalizeHist(gray)
    cv.imshow('equal_hist_demo',dst)


def clahe_demo(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    dst=clahe.apply(gray)
    #dst = cv.equalizeHist(gray)
    cv.imshow('equal_hist_demo', dst)
# 生成直方图
def create_rgbhist(img):
    h, w, c = img.shape
    rgbHist = np.zeros([16*16*16, 1], np.float32)
    bsize = 256 / 16
    for row in range(h):
        for clo in range(w):
            b = img[row, clo, 0]
            g = img[row, clo, 0]
            r = img[row, clo, 0]
            index = np.int(b/bsize)*16*16 + np.int(g/bsize)*16 + np.int(r/bsize)
            rgbHist[np.int(index), 0] += 1

    return rgbHist
# 直方图比较
def histCompare_demo(img1, img2):
    hist1 = create_rgbhist(img1)
    hist2 = create_rgbhist(img2)
    match1 = cv.compareHist(hist1, hist2, cv.HISTCMP_BHATTACHARYYA)
    match2 = cv.compareHist(hist1, hist2, cv.HISTCMP_CORREL)
    match3 = cv.compareHist(hist1, hist2, cv.HISTCMP_CHISQR)
    print(match1, match2, match3)

def hist2d_demo(img):
    hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)
    hist = cv.calcHist([img], [0,1], None, [180,256],[0,180,0,256])
    cv.imshow('hist2d',hist)


img = cv.imread('lena.png',flags=cv.IMREAD_GRAYSCALE)


img1 = cv.imread('1.jpg')
img2 = cv.imread('2.jpg')
img3 = cv.bitwise_and(img1,img1)

cv.namedWindow('lena',cv.WINDOW_AUTOSIZE)
cv.imshow('lena',img3)
cv.imshow('img1',img1)
#plot_demo(img)
#histogram_demo(img)
#equalHis_demo(img)
#clahe_demo(img)
#histCompare_demo(img1,img2)
k = cv.waitKey(0)
if k == 27:
    cv.destroyAllWindows()