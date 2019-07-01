import cv2 as cv

img1 = cv.imread('IMG_3873.jpg')
img2 = cv.imread('WechatIMG89.jpg')
img3 = cv.imread('1.jpg')
img4 = cv.imread('2.jpg')



img5 = cv.resize(img1,(1920,1080))
img6 = cv.resize(img2,(1920,1080))
print(img5.shape,img6.shape)

cv.imshow('bb',img1)
cv.imshow('xx',img2)
dst = cv.addWeighted(img5, 0.8, img6, 0.3, 0) # shape相同

cv.imshow('dst_img',dst)
cv.waitKey(0)
cv.destroyAllWindows()