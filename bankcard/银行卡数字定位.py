from cv2 import cv2 as cv
import copy
import math
import numpy as np
from imutils import contours
import imutils
from matplotlib import pyplot as plt

def horzonl(img,pixel_sum):
    width = img.shape[1]
    height = img.shape[0]
    white_image = np.zeros(img.shape, np.uint8)
    white_image = white_image + 255
    h = [0] * img.shape[0]
    for x in range(height):
        for y in range(width):
            #print(x,y)
            k = img.item(x, y)
            # 统计黑的个数
            if k == 0:
                h[x] += 1
    # 绘制垂直投影图
    for x in range(len(h)):
        for i in range(h[x]):
            # 把大于0的像素变成白
            # print(x,i)
            #print(width - i)
            white_image[x,width - i - 1] = 0
    plt.plot(range(height),h)
    return h


def cutImg(img,h):

    Thr = sum(h) / (img.shape[0])
    print(Thr)
    low = 0
    high = 0
    flag = 0
    for i,val in enumerate(h):
        print(i,val)
        if flag == 0:
            if val < Thr:
                print('val====',val)
                flag = 1
                low = i
                continue
        #找到一个小于阈值的位置 找下一个大于阈值的位置 如果位置之差小于50，则继续从找小于阈值的位置开始
        if flag == 1:
            if val > Thr:
                high = i
                if high - low < 30:
                    flag = 0
                    continue
                else:
                    break
    #return low, high
    img = img[low-15:high+15,:]
    return img

    # print(white_image)
    # cv.imshow('paintx1', white_image)
    # cv.imshow('thresh',img)
    #cv.waitKey(0)


    
def YUVCanny(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Y = gray[:, :, 0]
    # U = gray[:, :, 1]
    # V = gray[:, :, 2]
    cany_origin = cv.Canny(gray, 70, 80)
    # cany_u = cv.Canny(U, 40, 50)
    # cany_v = cv.Canny(V, 20, 30)
    kernel = np.ones((3, 3), np.uint8)
    # dilate_canny_u = cv.dilate(cany_u, kernel)
    # dilate_canny_v = cv.dilate(cany_v, kernel)
    # canny_number = cany_origin -  cany_v 
    cv.imshow('canny_origh', cany_origin)
    # cv.imshow('canny_number', canny_number)
    return cany_origin

def checksum(string):
    digits = list(map(int, string))
    odd_sum = sum(digits[-1::-2])
    even_sum = sum([sum(divmod(2 * d, 10)) for d in digits[-2::-2]])
    print(odd_sum + even_sum)
    return (odd_sum + even_sum) % 10  

def loacateNum(img):
    canny = YUVCanny(img)
    pixel_sum = np.sum(canny/255)
    h = horzonl(canny,pixel_sum)
    return cutImg(img,h)

if __name__ == "__main__":
    # img = cv.imread('bankcard/16.jpg')
    # num = loacateNum(img)
    # cv.imshow("Nmber",num)
    # plt.show()
    # k = cv.waitKey(0)
    # if k == 27:
    #     cv.destroyAllWindows()
    checksum('6222081209001537078')


