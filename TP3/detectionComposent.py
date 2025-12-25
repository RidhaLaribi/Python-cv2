import cv2
import numpy as  np

img=cv2.imread("img_1.png",0)
img1=cv2.imread("img_1.png")
_,binary= cv2.threshold(img,127,255,cv2.THRESH_BINARY)
l,label,s,c=cv2.connectedComponentsWithStats(binary,255-binary)




for i in range(l):
    x,y,w,h,area=s[i]
    # cx,cy=c[i]
    cv2.rectangle(img1, (x, y), (x + w, y + w), (255, 0, 0), 2)
cv2.imshow('fgh',img1)

cv2.waitKey(0)


# print([label,l,x,d])