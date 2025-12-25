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



import cv2
import numpy as np


im1 = cv2.imread("../Images/pepper.bmp")

h, w = im1.shape[:2]
newh = int(h / 2)
neww = int(w / 2)

im2 = np.zeros((newh, neww, 3), dtype=np.uint8)

ii = 0
for i in range(0, h - 1, 2):
    jj = 0
    for j in range(0, w - 1, 2):

        block = (
                        im1[i, j].astype(np.float32)
                        + im1[i + 1, j].astype(np.float32)
                        + im1[i, j + 1].astype(np.float32)
                        + im1[i + 1, j + 1].astype(np.float32)
                ) / 4.0

        im2[ii, jj] = block.astype(np.uint8)
        jj += 1
    ii += 1

im3 = manual(im1,int(h/2),int(w/2))
cv2.imshow("Original Image", im1)
cv2.imshow("(manuelle)", im3)
cv2.imshow("Image Réduite (manuelle)", im2)
cv2.waitKey(0)
cv2.destroyAllWindows()











