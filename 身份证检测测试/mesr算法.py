from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob

imgs = glob('身份证检测测试/image/ying.png')
imgs.sort()
i = 0
for img in imgs:
    i=i+1
    imgName = img.split('/')[-1]
    print(i,imgName)
    image = cv2.imread(img)
    #image = cv2.resize(image,(800,500),interpolation=cv2.INTER_CUBIC)
    mser  = cv2.MSER_create(_min_area=50,_max_area=300)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    regions, boxes = mser.detectRegions(gray)
    a = np.zeros(shape = gray.shape)
    for box in boxes:
        x, y, w, h = box
        a[y:y+h,x:x+w] = 255
        cv2.rectangle(image, (x,y),(x+w, y+h), (255, 0,0), 2)
    kernel = np.ones((2,25),np.uint8)
    dialtion = cv2.dilate(a,kernel,iterations = 1)
    cv2.imwrite('身份证检测测试/result/bin_' + imgName,dialtion)
    cv2.imwrite('身份证检测测试/result/' + imgName,image)

    