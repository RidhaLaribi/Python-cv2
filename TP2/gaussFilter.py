
import cv2
import numpy as np
def gaussianFilter(image, sigma, maskDiameter):
    H, W = image.shape[:2]
    radius = maskDiameter // 2
    kernel = cv2.getGaussianKernel(maskDiameter, sigma)
    kernel = kernel * kernel.T

    image = image.astype(np.float32)
    filtrdImg = np.zeros((H, W), np.float32)
    for iy in range(radius, H - radius + 1):
        for jx in range(radius, W - radius + 1):
           sumPixel = 0.0
           for ky in range(iy - radius, maskDiameter):
               for kx in range(jx - radius, maskDiameter):
                   sumPixel += image[iy + ky, jx + kx] * kernel[ky, kx]
           filtrdImg[iy, jx] = sumPixel
    filtrdImg = np.clip(filtrdImg, 0, 255).astype(np.uint8)
    return filtrdImg


im1 = cv2.imread("Images/cameraman.bmp",cv2.IMREAD_GRAYSCALE)
im=gaussianFilter(im1,10,5)
cv2.imshow("1fmoy",im1)
cv2.imshow("oy",im)
cv2.waitKey(0)
