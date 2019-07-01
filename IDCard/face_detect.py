from cv2 import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

def verticalProj(img):

    #print(img.shape)
    width = img.shape[1]
    height = img.shape[0]
    white_image = np.zeros(img.shape,np.uint8)
    white_image = white_image+255
    w = [0]*img.shape[1]

    x1 = range(len(w))
    #print(x1)
    for x in range(width):
        for y in range(height):
            #print(x,y)
            k = img.item(y, x)
            if k == 0:
                w[x] += 1

    # 绘制垂直投影图

    for x in range(len(w)):
        for i in range(w[x]):
            # 把大于0的像素变成白
            #print(x,i)
            #print(height-i)
            white_image[height-i-1, x] = 0

    #print(white_image)
    plt.plot(x1,w)
    plt.show()
    #cv.imshow('paintx', white_image)
    #cv.imshow('thresh',img)
    cv.waitKey(0)

def horzonl(img):

    width = img.shape[1]
    height = img.shape[0]
    white_image = np.zeros(img.shape, np.uint8)
    white_image = white_image + 255

    h = [0] * img.shape[0]

    for x in range(height):
        for y in range(width):
            #print(x,y)
            k = img.item(x, y)
            if k == 0:
                h[x] += 1

    # 绘制垂直投影图

    for x in range(len(h)):
        for i in range(h[x]):
            # 把大于0的像素变成白
            # print(x,i)
            print(width - i)
            white_image[x,width - i - 1] = 0

    print(white_image)
    cv.imshow('paintx1', white_image)
    # cv.imshow('thresh',img)
    cv.waitKey(0)

def HouLine(img,canny_img):
    lines = cv.HoughLines(canny_img, 1, np.pi / 180, 150)
    print(lines)
    for i in range(len(lines)):
        for rho, theta in lines[i]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv.imshow('houp',img)

def HouLinep(img,canny_img)  :
    minLineLength = 100
    maxLineGap = 10
    lines = cv.HoughLinesP(canny_img, 1, np.pi / 180, 120, minLineLength, maxLineGap)
    for x1, y1, x2, y2 in lines[0]:
        cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv.imshow('houp',img)




if __name__ == "__main__":
    img = cv.imread('sfz1.jpeg')
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    _, thresh_sfz = cv.threshold(gray, 80, 255, cv.THRESH_BINARY)

    verticalProj(thresh_sfz)
    # horzonl(thresh_sfz)

    cany_sfz = cv.Canny(thresh_sfz, 180, 255)
    # contours, hierarchy = cv.findContours(cany_sfz, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # cv.drawContours(img, contours, -1, (0, 255, 0), 3)
    #HouLinep(img,cany_sfz)


    # cv.imshow('graysfz', gray)
    # cv.imshow('thresh_sfz', thresh_sfz)



    # cv.imshow('counters', img)
    cv.imshow('canny',cany_sfz)
    k = cv.waitKey(0)
    if k == 27:
        cv.destroyAllWindows()







