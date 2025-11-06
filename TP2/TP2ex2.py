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

def calculate_psnr(im1, im2):
    mse = np.mean((im1.astype(np.float32) - im2.astype(np.float32)) ** 2)
    if mse == 0:
        return float("inf")  # Images identical
    R = 255.0
    psnr = 10 * np.log10((R * R) / mse)
    return psnr



# ---- PSNR using OpenCV ----
psnr_im1_im2 = cv2.PSNR(im1, im2)
psnr_im1_imf2 = cv2.PSNR(im1, imf2)
psnr_im1_imfm2 = cv2.PSNR(im1, imf22)
psnr_im1_im3 = cv2.PSNR(im1, im3)
psnr_im1_imf3 = cv2.PSNR(im1, imf3)
psnr_im1_imfm3 = cv2.PSNR(im1, imf33)

print("----- PSNR with OpenCV -----")
print("PSNR(im1, im2) =", psnr_im1_im2)
print("PSNR(im1, imf2) =", psnr_im1_imf2)
print("PSNR(im1, imfm2) =", psnr_im1_imfm2)
print("PSNR(im1, im3) =", psnr_im1_im3)
print("PSNR(im1, imf3) =", psnr_im1_imf3)
print("PSNR(im1, imfm3) =", psnr_im1_imfm3)

# ---- PSNR using our own function ----
print("\n----- PSNR with custom function -----")
print("PSNR(im1, im2) =", calculate_psnr(im1, im2))
print("PSNR(im1, imf2) =", calculate_psnr(im1, imf2))
print("PSNR(im1, imfm2) =", calculate_psnr(im1, imf33))
print("PSNR(im1, im3) =", calculate_psnr(im1, im3))
print("PSNR(im1, imf3) =", calculate_psnr(im1, imf3))
print("PSNR(im1, imfm2) =", calculate_psnr(im1, imf33))



import time


results = []

#   MOY FILTERS
for k in [3, 5, 7]:
    start = time.time()
    f = cv2.blur(im2, (k, k))
    t = time.time() - start
    results.append(("Moy", f"{k}x{k}", "im2", cv2.PSNR(im1, f), t))

    start = time.time()
    f = cv2.blur(im3, (k, k))
    t = time.time() - start
    results.append(("Moy", f"{k}x{k}", "im3", cv2.PSNR(im1, f), t))


# GAUSSIAN FILTERS
for k in [3, 5]:
    for sigma in [2, 1.5, 1, 0.5]:
        start = time.time()
        f = cv2.GaussianBlur(im2, (k, k), sigma)
        t = time.time() - start
        results.append(("Gaussian", f"{k}x{k}, σ={sigma}", "im2", cv2.PSNR(im1, f), t))

        start = time.time()
        f = cv2.GaussianBlur(im3, (k, k), sigma)
        t = time.time() - start
        results.append(("Gaussian", f"{k}x{k}, σ={sigma}", "im3", cv2.PSNR(im1, f), t))


# MEDIAN FILTERS
for k in [3, 5, 7]:
    start = time.time()
    f = cv2.medianBlur(im2, k)
    t = time.time() - start
    results.append(("Median", f"{k}x{k}", "im2", cv2.PSNR(im1, f), t))

    start = time.time()
    f = cv2.medianBlur(im3, k)
    t = time.time() - start
    results.append(("Median", f"{k}x{k}", "im3", cv2.PSNR(im1, f), t))


# BILATERAL FILTER
start = time.time()
f = cv2.bilateralFilter(im2, 7, 40, 40)
t = time.time() - start
results.append(("Bilateral", "d=7", "im2", cv2.PSNR(im1, f), t))

start = time.time()
f = cv2.bilateralFilter(im3, 7, 40, 40)
t = time.time() - start
results.append(("Bilateral", "d=7", "im3", cv2.PSNR(im1, f), t))

print("\n---- RESULTS ----")
print("Filter\t\tParams\t\tImage\tPSNR(dB)\tTime(s)")
for r in results:
    print(f"{r[0]:10} {r[1]:15} {r[2]:5} {r[3]:8.3f}   {r[4]:.6f}")