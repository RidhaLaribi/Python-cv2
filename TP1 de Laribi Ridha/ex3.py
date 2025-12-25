import cv2
import numpy
import numpy as np

img=cv2.imread("Images/barbara.bmp",cv2.IMREAD_GRAYSCALE)

_,img1=cv2.threshold(img,80,255,cv2.THRESH_BINARY)
img2=np.zeros_like(img)
h,w=img.shape

for i in range(h):
    for j in range(w):
        if img[i,j]>80:img2[i,j]=255
        else:img2[i,j]=0


cv2.imshow("img",img)

cv2.imshow("img1",img1)

cv2.imshow("img2",img2)
cv2.waitKey(0)