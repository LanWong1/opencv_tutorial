import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

def resize_img(img):
    # fx 横向倍数  fy 纵向倍数
    dst = cv.resize(img, None, fx=2, fy=1.5, interpolation=cv.INTER_CUBIC)
    cv.imshow('resize',dst)
# move
def translation_demo(img):
    # row = height  cols = width
    rows, cols = img.shape[0],img.shape[1]
    # 移动矩阵 x = 100, y= 100
    M = np.float32([[1,0,100],[0,1,100]])
    dst = cv.warpAffine(img, M,(rows,cols))
    cv.imshow('wrapaffine',dst)

def rotation_demo(img):
    rows, cols = img.shape[0:2]#,img.shape[1]
    M = cv.getRotationMatrix2D(((cols-1)/2.0, (rows-1)/2.0), 90, 0.5)
    dst = cv.warpAffine(img,M,(rows,cols))
    cv.imshow('rotation',dst)

def affine_transform_demo(img):
    rows,cols, ch = img.shape
    pts1 = np.float32([[50,50],[200,50],[50,200]])
    pts2 = np.float32([[10,100],[200,50],[100,250]])
    M = cv.getAffineTransform(pts1, pts2)
    dst = cv.warpAffine(img, M, (cols, rows))
    #plt.subplot(121), plt.imshow(img),plt.title('Input')
    cv.imshow('tranform', dst)

def perspective_transformation(img):
    rows, cols, ch = img.shape
    pts1 = np.float32([[56, 65], [368, 52], [28, 387],[389,390]])
    pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
    M= cv.getPerspectiveTransform(pts1, pts2)
    dst = cv.warpPerspective(img, M, (300, 300))
    cv.imshow('perspective',dst)


img = cv.imread('lena.png')
cv.namedWindow('lena',cv.WINDOW_AUTOSIZE)
cv.imshow('lena',img)
#resize_img(img)
#translation_demo(img)
#rotation_demo(img)
#affine_transform_demo(img)
perspective_transformation(img)
k = cv.waitKey(0)
if k == 27:
    cv.destroyAllWindows()