import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(threshold=np.inf)




img = cv.imread('ID_Card_for_test.png')
w = int(img.shape[1]/2)
img_half = img[:,:w,:]
img_gray = cv.cvtColor(img_half,cv.COLOR_BGR2GRAY)
ret,img_th = cv.threshold(img_gray,110,255,cv.THRESH_BINARY)
(h,w) = img_th.shape
thresh_count = [0 for z in range(0,h)]
for i in range(0,h):
    for j in range(0,w):
        if img_th[i,j] == 0:
            thresh_count[i] =thresh_count[i] + 1

cout = np.arange(0,h)
black_cout = np.array(thresh_count)
black_cout_mean = np.mean(black_cout)
inx = []
for index, black_cnt in enumerate(black_cout):
    if black_cnt > 10:
        inx.append(index)
befor = 0
cut_range = []
for i in range(0,len(inx)-1):
    if inx[i+1] - inx[i]>1:
        end = i
        cut_range.append([inx[befor], inx[end]])
        befor = i+1
    if i == len(inx) - 2:
        end = i+1
        cut_range.append([inx[befor], inx[end]])
print(cut_range)

i=0
for rng in cut_range:
    img_cut = img[(rng[0]-5):(rng[1]+5),:,:]
    i = i + 1
    cv.imshow('thresh' + str(i), img_cut)
    cv.waitKey(10000)




#print(inx[befor],inx[end])
#print(inx)
# plt.plot(cout,black_cout)
# plt.show()
# cv.imshow('gray',img_gray)
# cv.imshow('thresh',img_th)
# cv.waitKey(0)