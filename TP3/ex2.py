import cv2
import numpy as np

img = cv2.imread("img_1.png",0)
gr=np.zeros_like(img)

cv2.imshow("efda",img)
cv2.waitKey(0)

h,w=img.shape

for i in range(1,h,3):
    for j in range(1,w,3):
        # print(img[i-1:i+2,j-1:j+2])
        gr[i-1:i+2,j-1:j+2]=img[i-1:i+2,j-1:j+2]*[[-1,0,1],[-1,0,1],[-1,0,1]]
cv2.imshow("efda",gr)
cv2.waitKey(0)
