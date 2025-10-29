import numpy
import cv2

im1 = cv2.imread("Images/pepper.bmp")

h, w = im1.shape[:2]

im2 = cv2.resize(im1, (w // 2, h // 2))

im3 = im1[30:150, 200:400]

im4 = im1.copy()
cv2.rectangle(im4, (200, 30), (400, 150), (255, 0, 0), 2)
cv2.putText(im4, "My rectangle", (200, 20), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 255, 0), 2)

im5 = cv2.rotate(im1, cv2.ROTATE_90_CLOCKWISE)

im6 = cv2.GaussianBlur(im1, (9, 9), 0)

cv2.imshow("im1", im1)
cv2.imshow("im2", im2)
cv2.imshow("im3", im3)
cv2.imshow("im4", im4)
cv2.imshow("im5", im5)
cv2.imshow("im6", im6)
cv2.waitKey(0)
