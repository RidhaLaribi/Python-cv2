import cv2
import numpy as np


def filterM(im,r):
    hm=2*r+1
    t=hm**2
    imflt=im.copy()     #.astype(np.float32)
    # img=im.astype(np.float32)
    h,w=im.shape[:2]


    for i in range(r,h-r):
        for j in range(r,w-r):
            sum=0
            reg= im[i-r:i+r+1,j-r:j+r+1]
            mo1=np.mean(reg)
            for k in range(-r+1,r):
                for z in range( -r+1,  r ):
                    # moy=np.mean(im[k:i+k,z:j+z])
                    sum+=int(im[i+k,j+z])
            moy=sum/t
            imflt[i,j]=moy
    for i in range(r, h - r):
        for j in range(r, w - r):
            sum = 0
            for k in range(-r+1 , r-1):
                for z in range(-r , 0):
                    sum+=int(im[i+k,j+z])
            sum += int(im[i,j])
            moy = sum / t
            imflt[i, j] = moy

    for i in range(r, h - r):
        for j in range(r, w - r):
            sum = 0
            for k in range(-r+1 , r-1):
                for z in range(0 , r):
                    sum+=int(im[i+k,j+z])
            sum += int(im[i,j])
            moy = sum / t
            imflt[i, j] = moy
    for i in range(r, h - r):
        for j in range(r, w - r):
            sum = 0
            for k in range(-r , 0):
                for z in range(-r+1 , r-1):
                    sum+=int(im[i+k,j+z])
            sum += int(im[i,j])
            moy = sum / t
            imflt[i, j] = moy

    for i in range(r, h - r):
        for j in range(r, w - r):
            sum = 0
            for k in range(0 , r):
                for z in range(-r+1 , r-1):
                    sum+=int(im[i+k,j+z])
            sum += int(im[i,j])
            moy = sum / t
            imflt[i, j] = moy


    # for i in range(r,h-r):
    #     for j in range(r,w-r):
    #         sum=0
    #         reg= im[i-r:i+r+1,j-r:j+r+1]
    #         mo1=np.mean(reg)
    #         for k in range(-r,r+1):
    #             for z in range( - r,  r+1 ):
    #                 # moy=np.mean(im[k:i+k,z:j+z])
    #                 sum+=int(im[i+k,j+z])
    #         moy=sum/t
    #         imflt[i,j]=moy
    return imflt


im1=cv2.imread('Images/cameraman.bmp',cv2.IMREAD_GRAYSCALE)
im2=cv2.imread('Images/cameraman_bruit_gauss_sig0_001.bmp',cv2.IMREAD_GRAYSCALE)
im3=cv2.imread('Images/cameraman_bruit_sel_poivre_p_10.bmp',cv2.IMREAD_GRAYSCALE)
# im1=cv2.imread('../Images/cameraman.bmp',cv2.IMREAD_GRAYSCALE)
# im1=cv2.imread('../Images/cameraman.bmp',cv2.IMREAD_GRAYSCALE)

# cv2.imshow("1",im1)
# cv2.imshow("2",im2)
# cv2.imshow("3",im3)

imf1=filterM(im1,2)
cv2.imshow("1fmoy",imf1)
cv2.waitKey(0)

