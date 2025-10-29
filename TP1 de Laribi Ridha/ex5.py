# TP1_Exo5.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

im1 = cv2.imread("Images/cameraman.bmp", cv2.IMREAD_GRAYSCALE)

im2 = cv2.imread("Images/pepper.bmp")

#pour l'image en niveaux de gris
hist1 = cv2.calcHist([im1], [0], None, [256], [0, 256])

#pour l'image couleur

hist2_B = cv2.calcHist([im2], [0], None, [256], [0, 256])
hist2_G = cv2.calcHist([im2], [1], None, [256], [0, 256])
hist2_R = cv2.calcHist([im2], [2], None, [256], [0, 256])

plt.figure(" Histograms")


plt.subplot(2, 1, 1)
plt.plot(hist1, color='black')
plt.title("im1")
plt.xlabel("Gray Level")
plt.ylabel("Frequency")


plt.subplot(2, 1, 2)
plt.plot(hist2_B, color='blue')
plt.plot(hist2_G, color='green')
plt.plot(hist2_R, color='red')
plt.title("im2")
plt.xlabel("RGB Level")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

###############################################################################################################
#manuel
#pour l'image en niveaux de gris
histp1 = np.zeros(256, dtype=int)
for pixel in im1.ravel():
    histp1[pixel] += 1

#pour l'image couleur
histp2_B = np.zeros(256, dtype=int)
histp2_G = np.zeros(256, dtype=int)
histp2_R = np.zeros(256, dtype=int)

for row in im2:
    for pixel in row:
        histp2_B[pixel[0]] += 1
        histp2_G[pixel[1]] += 1
        histp2_R[pixel[2]] += 1





plt.figure(" Histograms")


plt.subplot(2, 1, 1)
plt.plot(hist1, color='black')
plt.title("im1")
plt.xlabel("Gray Level")
plt.ylabel("Frequency")


plt.subplot(2, 1, 2)
plt.plot(hist2_B, color='blue')
plt.plot(hist2_G, color='green')
plt.plot(hist2_R, color='red')
plt.title("im2")
plt.xlabel(" RGB Level")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()









































