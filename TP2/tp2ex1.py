import cv2
import numpy as np
import random

# -------------------- 1) Lecture de l'image --------------------
im1 = cv2.imread('../Images/cameraman.bmp', cv2.IMREAD_GRAYSCALE)

# -------------------- 2) Bruit gaussien additif --------------------

gaussian_noise = np.random.normal(0, 10, im1.shape)

print(gaussian_noise)

im2 = im1 + gaussian_noise
im2 = np.clip(im2, 0, 255).astype(np.uint8)

cv2.imshow("Image originale", im1)
cv2.imshow("Image avec bruit gaussien", im2)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('../Images/cameraman_bruit_gauss_sig0_001.bmp', im2)


# -------------------- 3) Bruit sel et poivre --------------------
def sp_noise(image, prob):
    output = image.copy()

    if len(image.shape) == 2:
        black = 0
        white = 255
    else:
        colorspace = image.shape[2]
        if colorspace == 3:
            black = [0, 0, 0]
            white = [255, 255, 255]
        else:
            black = 0
            white = 255

    probs = np.random.random(output.shape[:2])

    output[probs < (prob / 2)] = black
    print("prb ",[probs < (prob / 2)])

    output[probs > 1 - (prob / 2)] = white

    return output
#
# def sp_noise(image, prob):
#
#     output = np.copy(image)
#     total_pixels = image.size
#     num_salt = np.ceil(prob * total_pixels / 2)
#     print(num_salt)
#     num_pepper = np.ceil(prob * total_pixels / 2)
#
#
#     coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
#     output[coords[0], coords[1]] = 255
#
#     # Bruit "poivre" (pixels noirs)
#     coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
#     output[coords[0], coords[1]] = 0
#
#     return output


p = 0.1
im3 = sp_noise(im1, p)

# Affichage
cv2.imshow("Image originale", im1)
cv2.imshow(f"Image bruitée )", im3)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('../Images/cameraman_bruit_sel_poivre_p_10.bmp', im3)

# -------------------- 4) Comparaison des deux types de bruit --------------------
cv2.imshow("Originale", im1)
cv2.imshow("Bruit Gaussien", im2)
cv2.imshow("Bruit Sel et Poivre", im3)
cv2.waitKey(0)
cv2.destroyAllWindows()


