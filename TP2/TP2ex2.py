import cv2
import numpy as np

im1=cv2.imread('Images/cameraman.bmp',cv2.IMREAD_GRAYSCALE)
im2=cv2.imread('Images/cameraman_bruit_gauss_sig0_001.bmp',cv2.IMREAD_GRAYSCALE)
im3=cv2.imread('Images/cameraman_bruit_sel_poivre_p_10.bmp',cv2.IMREAD_GRAYSCALE)
# im1=cv2.imread('../Images/cameraman.bmp',cv2.IMREAD_GRAYSCALE)
# im1=cv2.imread('../Images/cameraman.bmp',cv2.IMREAD_GRAYSCALE)

cv2.imshow("1",im1)
cv2.imshow("2",im2)
cv2.imshow("3",im3)


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


            for k in range(-r,r+1):
                for z in range( - r,  r+1 ):
                    # moy=np.mean(im[k:i+k,z:j+z])
                    sum+=int(im[i+k,j+z])


            moy=sum/t

            imflt[i,j]=moy

            # # for p in range(k - r, k + r+1):
            #     for m in range(z - r, z + r+1):

    return imflt

def filtermed(img,t):
    imf=img.copy()
    if t%2==0:
        return

    r=t//2
    h,w=img.shape

    for i in range(r,h-r):
        for j in range(r,h-r):
            c=[]
            # c=img[i-r:i+r,j-r:j+r]

            for k in range(i-r,i+r):
                for z in range(j-r,j+r):
                    c.append((img[k,z]))
            imf[i,j]=np.median(c)

    return imf







imf1=filterM(im1,2)
cv2.imshow("1fmoy",imf1)
imf11=filtermed(im1,9)
cv2.imshow("1fmed",imf11)

imf2=filterM(im2,2)
cv2.imshow("2fmoy",imf2)
imf22=filtermed(im2,9)
cv2.imshow("2fmed",imf22)

imf3=filterM(im1,2)
cv2.imshow("3fmoy",imf3)
imf33=filtermed(im1,9)
cv2.imshow("3fmed",imf33)



#
# imf2=cv2.blur(im1,(5,5))
# cv2.imshow("boit noir",imf2)




cv2.waitKey(0)
cv2.destroyAllWindows()