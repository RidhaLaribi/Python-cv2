import cv2
import numpy as np


im1 = cv2.imread("Images/pepper.bmp")

h, w = im1.shape[:2]
print(f"image : hauteur = {h}, largeur = {w}")



def manual(img,nw,nh):
    oldh,oldw=img.shape[:2]
    resized=np.zeros((nh,nw,3),dtype=np.uint8)
    xrat=oldw/nw
    yrat=oldh/nh
    for i in range(nh):
        for j in range(nw):
            x= int(j*xrat)
            y=int(i*yrat)
            resized[i,j]=img[y,x]
    return resized

im2 = manual(im1,int(h/2),int(w/2))
cv2.imshow("Original Image", im1)
cv2.imshow("Image Réduite (manuelle)", im2)
cv2.waitKey(0)
cv2.destroyAllWindows()