from cv2 import cv2 as cv
import numpy as np
import copy
from matplotlib import pyplot as plt

# 直线检测很给力 比轮廓检测更好 更简单方便 可用于银行卡切边
def lsd_Detct(img):
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    print(gray)
    gray_mean = np.mean(gray)
    _,thresh= cv.threshold(gray,gray_mean-30,255,cv.THRESH_BINARY_INV)
    detector = cv.createLineSegmentDetector(0)
    lines,width,prec,nfa=detector.detect(thresh)
    line_lens = []
    for line in lines[:,0]:
        x1,y1,x2,y2 = line

        a = pow((y2-y1),2) + pow((x2-x1),2)
        line_len = pow(a,0.5)
        line_lens.append(line_len)
    line_lenths = line_lens.copy()
    line_lens.sort()
    index = []

    lin_len_top4 = line_lens[-4:]
    for item in lin_len_top4:
        index.append(line_lenths.index(item))


    detector.drawSegments(img,lines[index])
    points = []
    point_x = []
    #detector.drawSegments(img, lines[2])
    point1 = (pt1_x, pt1_y) = crossPoint(lines[index[0]][0], lines[index[2]][0])
    point_x.append(pt1_x)
    points.append(point1)

    point2 = (pt2_x, pt2_y) = crossPoint(lines[index[0]][0], lines[index[3]][0])
    points.append(point2)
    point_x.append(pt2_x)

    point3 = (pt3_x, pt3_y) = crossPoint(lines[index[1]][0], lines[index[2]][0])
    points.append(point3)
    point_x.append(pt3_x)
    point4 = (pt4_x, pt4_y) = crossPoint(lines[index[1]][0], lines[index[3]][0])
    points.append(point4)
    point_x.append(pt4_x)
    point_x_unsort = point_x.copy()
    point_x.sort()

    index_min_x1 = point_x_unsort.index(point_x[0])
    index_min_x2 = point_x_unsort.index(point_x[1])
    index_min_x3 = point_x_unsort.index(point_x[2])
    index_min_x4 = point_x_unsort.index(point_x[3])

    src_points = []
    if points[index_min_x1][1] < points[index_min_x2][1]:
        point_left_top = points[index_min_x1]
        point_left_bottom = points[index_min_x2]
    else:
        point_left_top = points[index_min_x2]
        point_left_bottom = points[index_min_x1]

    if points[index_min_x3][1] < points[index_min_x4][1]:
        point_right_top = points[index_min_x3]
        point_right_bottom = points[index_min_x4]
    else:
        point_right_top = points[index_min_x4]
        point_right_bottom = points[index_min_x3]
    #for point in points:
    cv.circle(img, point_left_top, 5, (0, 0, 255))
    cv.circle(img, point_left_bottom, 5, (0, 255, 0))
    cv.circle(img, point_right_top, 5, (255, 0, 0))
    cv.circle(img, point_right_bottom, 5, (255, 255, 255))

    src_points.append(point_left_top)
    src_points.append(point_right_top)
    src_points.append(point_right_bottom)
    src_points.append(point_left_bottom)
    for item in src_points:
        list(item)
    src = np.float32([point_left_top, point_right_top,point_right_bottom,point_left_bottom])
    dst_points = np.float32([[0,0],[300,0],[300,190],[0,190]])
    M = cv.getPerspectiveTransform(src,dst_points)
    dst = cv.warpPerspective(img,M,(300,190))
    cv.rectangle(img,point_left_bottom,point_right_top,(0,255,0))

    #img = img[int(pt1_y):int(pt2_y),int(pt3_x):int(pt1_x),:]

    cv.imshow('dst',dst)
    cv.imshow('lines',img)
    cv.imshow('thesh',thresh)
    cv.imshow('gray',gray)

def crossPoint(line1,line2):
    X1 = line1[2] - line1[0]
    Y1 = line1[3] - line1[1]
    X2 = line2[2] - line2[0]
    Y2 = line2[3] - line2[1]
    D = Y1 * X2 - Y2 * X1
    X21 = line2[0] - line1[0]
    Y21 = line2[1] - line1[1]
    if D == 0:
        return 0
    pt_x = (X1 * X2 * Y21 + Y1 * X2 * line1[0] - Y2 * X1 * line2[0]) / D
    pt_y = -(Y1 * Y2 * X21 + X1 * Y2 * line1[1] - X2 * Y1 * line2[1]) / D
    # if ((abs(pt_x - line1[0] - X1 / 2) <= abs(X1 / 2)) and (abs(pt_y - line1[1] - Y1 / 2) <= abs(Y1 / 2)) and (abs(pt_x - line2[0] - X2 / 2) <= abs(X2 / 2)) and (abs(pt_y - line2[1]- Y2 / 2) <= abs(Y2 / 2))):
    #     return pt_x, pt_y
    return pt_x, pt_y



def grey_scale(image):
    rows,cols = image.shape
    flat_gray = image.reshape((cols * rows,)).tolist()
    A = min(flat_gray)
    B = max(flat_gray)
    print('A = %d,B = %d' %(A,B))
    output = np.uint8(255 / (B - A) * (image - A) + 0.5)
    return output


def abstructCard(img):
   gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
   print(gray)
   gray_mean = np.mean(gray)
   _, thresh = cv.threshold(gray, gray_mean-10, 255, cv.THRESH_BINARY_INV)
   img1 =  img.copy()
   #cv.imshow('gray',gray)
   #cv.imshow('thresh', thresh)
   cv.imshow('th3',thresh)
   canny = cv.Canny(thresh,100,150)
   #cv.imshow('canny',canny)
   image, coutous, hierarchy= cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
   cv.drawContours(img,coutous,-1,(0,0,255),5)
   coutous_Area = []
   for item in coutous:
       #print('item shaoe = ',item.shape)
       coutous_Area.append(cv.contourArea(item))
   index = coutous_Area.index(max(coutous_Area))
   rec_area = 0
   x1 = 0
   y1 = 0
   w1 = 0
   h1 = 0
   for i,item in enumerate(coutous_Area):
       if(item >200):
           epsilon = 0.1 * cv.arcLength(coutous[i], True)
           approx = cv.approxPolyDP(coutous[i], epsilon, True)
           cv.drawContours(img, coutous, i, (0, 255, 0), 3)
           x, y, w, h = cv.boundingRect(coutous[i])
           area = w*h
           if(rec_area < area):
               rec_area = area
               x1 = x
               y1 = y
               w1 = w
               h1 = h
   rec = cv.rectangle(img, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)
   #cv.imshow('rectangle',rec)
   result = img1[y1:y1+h1,x1:x1+w1,:]

   cv.imshow("card",result)
   cv.imshow('img',img)


def floodFill(img):
    cv.floodFill(img)

if __name__ == "__main__":

    img = cv.imread('27.jpg')
    blur = cv.bilateralFilter(img, 9, 75, 75)
    abstructCard(blur)
    #lsd_Detct(blur)
    k = cv.waitKey(0)

