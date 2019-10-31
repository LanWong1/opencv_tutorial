import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from glob import glob



np.set_printoptions(threshold=np.inf)

def rotate_N_90(img_orign,n):
    img = img_orign.copy()
    for i in range(n):
        img1 = cv.transpose(img)
        img2 = cv.flip(img1, 1)
        img = img2
    return img2


def vertical(img):
    kernel = np.ones((5,5),np.uint8)


    img1 = img
    w = int(img.shape[1])
    h = int(img.shape[0])
    # img = img[:,int(0.2*w):int(0.6*w),:]
    img1 = cv.GaussianBlur(img,(5,5),0)
    img_gray = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)


    img_cany = cv.Canny(img_gray,80,150)

    # detect(img_cany)
    # cv.imshow('img_canny1', img_cany)
    # cv.waitKey(500)

    img_cany = cv.morphologyEx(img_cany,cv.MORPH_CLOSE, kernel)
    cv.imshow('img_canny',img_cany)
    cv.waitKey(500)

    (h,w) = img_cany.shape
    thresh_count = [0 for z in range(0,h)]
    for i in range(0,h):
        for j in range(0,w):
            if img_cany[i,j] == 255:
                thresh_count[i] =thresh_count[i] + 1


    cout = np.arange(0,h)
    black_cout = np.array(thresh_count)
    print(black_cout)
    cut_range = []



    inx = []
    for index, black_cnt in enumerate(black_cout):
        if black_cnt > 5:
            inx.append(index)
    print("index",inx)
    befor = 0
    #cut_range = []
    for i in range(0,len(inx)-1):
        if inx[i+1] - inx[i]>5:
            end = i
            if end - befor > 15:
                cut_range.append([inx[befor], inx[end]])
            befor = i+1

        if i == len(inx) - 2:
            end = i+1
            if end - befor > 15:
                cut_range.append([inx[befor], inx[end]])
    if len(cut_range) < 4 :
        cut_range = []

    print("cut range==========", cut_range)

    i=0
    for rng in cut_range:
        img_cut = img1[(rng[0]):(rng[1]),:,:]
        i = i + 1
        cv.imshow('thresh' + str(i), img_cut)
        cv.waitKey(500)

    plt.plot(cout, black_cout)
    plt.show()


# coding:utf8




def preprocess(gray,name):
    # 1. Sobel算子，x方向求梯度
    canny = cv.Canny(gray, 125,180)
    # 2. 二值化
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_OTSU + cv.THRESH_BINARY_INV)

    # 3. 膨胀和腐蚀操作的核函数
    erio_elem = cv.getStructuringElement(cv.MORPH_RECT, (15, 5))
    diala_elem = cv.getStructuringElement(cv.MORPH_RECT, (11,3))

    # 4. 膨胀一次，让轮廓突出
    dilation = cv.dilate(canny, diala_elem, iterations=1)
    # 5. 腐蚀一次，去掉细节，如表格线等。注意这里去掉的是竖直的线
    erosion = cv.erode(dilation, erio_elem, iterations=1)
    # 6. 再次膨胀，让轮廓明显一些
    #dilation2 = cv.dilate(erosion, diala_elem, iterations=3)
    # 7. 存储中间图片
    # cv.imwrite("binary.png", binary)
    # cv.imwrite("dilation.png", dilation)
    name = name.split('/')[len(name.split('/'))-1]
    cv.imwrite('result/'+name, dilation)
    #cv.imwrite("dilation2.png", dilation2)
    cv.imshow('canny', canny)
    cv.imshow('dilation',dilation)
    cv.imshow('erosion',erosion)
    #cv.imshow('dilation1', dilation2)
    cv.waitKey(500)
    #cv.destroyAllWindows()
    return dilation


def findTextRegion(img):
    region = []
    # 1. 查找轮廓
    image, contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    print(contours)
    # 2. 筛选那些面积小的
    for i in range(len(contours)):
        cnt = contours[i]
        # 计算该轮廓的面积
        area = cv.contourArea(cnt)

        # 面积小的都筛选掉
        # if (area < 1500):
        #     continue

        # 轮廓近似，作用很小
        epsilon = 0.001 * cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, epsilon, True)

        # 找到最小的矩形，该矩形可能有方向
        rect = cv.minAreaRect(cnt)
        # print("rect is: ")
        # print(rect)

        # box是四个点的坐标
        box = cv.boxPoints(rect)
        box = np.int0(box)

        # 计算高和宽
        height = abs(box[0][1] - box[2][1])

        width = abs(box[0][0] - box[2][0])
        print(width)
        if width < 50:
            continue
        # 筛选那些太细的矩形，留下扁的
        if (height > width * 1.2):
            continue

        region.append(box)
    print(region)
    return region


def detect(img, name):
    # 1.  转化成灰度图

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 2. 形态学变换的预处理，得到可以查找矩形的图片
    dilation = preprocess(gray,name)
    # 3. 查找和筛选文字区域
    region = findTextRegion(dilation)
    # 4. 用绿线画出这些找到的轮廓
    for box in region:
        cv.drawContours(img, [box], 0, (0, 255, 0), 2)
        cv.namedWindow("img", cv.WINDOW_NORMAL)
    cv.imshow("img", img)
    # 带轮廓的图片
    #cv.imwrite('result/'+name, img)
    cv.waitKey(0)
    #cv.destroyAllWindows()

def bentch_test():
    all_imgs = glob("/Users/ianwong/Documents/Sunyard/Sunyard_Git/Sunyard/身份证识别/testImg/*.jpg")  # 测试图片目录
    for img in all_imgs:
        print(img)
        image = cv.imread(img)
        image = cv.resize(image, (800, 500), interpolation=cv.INTER_CUBIC)
        detect(image, img)



if __name__ == '__main__':

    image = cv.imread('10.jpg')
    #image = cv.resize(image, (800, 500), interpolation=cv.INTER_CUBIC)
    detect(image, '20.jpg')
    #vertical(img)053409_14727_6A
