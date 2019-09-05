import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(threshold=np.inf)






def rotate_N_90(img_orign,n):
    img = img_orign.copy()

    for i in range(n):
        img1 = cv.transpose(img)
        img2 = cv.flip(img1, 1)
        img = img2
    return img2


def vertical(img):
    w = int(img.shape[1])
    img_half = img[:,:int(0.5*w),:]
    img1 = cv.GaussianBlur(img,(5,5),0)
    img_gray = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
    img_cany = cv.Canny(img_gray,80,150)
    #ret,img_th = cv.threshold(img_gray,110,255,cv.THRESH_BINARY)
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
    #black_cout_mean = np.mean(black_cout)
    inx = []
    for index, black_cnt in enumerate(black_cout):
        if black_cnt > 10:
            inx.append(index)
    print("index",inx)
    befor = 0
    cut_range = []
    for i in range(0,len(inx)-1):
        if inx[i+1] - inx[i]>5:
            end = i
            if end - befor > 10:
                cut_range.append([inx[befor], inx[end]])
            befor = i+1
        if i == len(inx) - 2:
            end = i+1
            if end - befor > 10:
                cut_range.append([inx[befor], inx[end]])
    print("cut range==========",cut_range)
    if len(cut_range) < 4 :
        cut_range = []
    i=0
    black_cout_mean = []
    for rng in cut_range:
        ran_list = range(rng[0],rng[1])
        np.sum(black_cout[ran_list])
        black_cout_mean.append(np.mean(black_cout[ran_list]))
        img_cut = img[(rng[0]):(rng[1]),:,:]
        i = i + 1
        cv.imshow('thresh' + str(i), img_cut)
        cv.waitKey(500)

    #print('dddddddd',black_cout_mean)
    #print(inx[befor],inx[end])
    #print(inx)
    plt.plot(cout,black_cout)
    plt.show()
    # cv.imshow('gray',img_gray)
    # cv.imshow('thresh',img_th)
    # cv.waitKey(0)

if __name__ == '__main__':
    img = cv.imread('2.jpg')
    vertical(img)
    #img2 = rotate_N_90(img,2)
    # cv.imshow('origin', img)
    # cv.imshow('rotate', img2)
    # cv.imshow('flib',img3)
    #cv.imwrite('2.jpg',img2)
    #cv.waitKey(0)