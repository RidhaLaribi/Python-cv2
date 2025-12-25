import cv2
import numpy as np
import math

def sobel_edge_detection(im, sb=30, sh=70):
    im = im.astype(np.float32)
    Gx_kernel = np.array([[-1, 0, 1],
                          [-2, 0, 2],
                          [-1, 0, 1]])
    Gy_kernel = np.array([[1, 2, 1],
                          [0, 0, 0],
                          [-1, -2, -1]])

    Gx = cv2.filter2D(im, -1, Gx_kernel)
    Gy = cv2.filter2D(im, -1, Gy_kernel)

    # Amplitude et orientation
    G = np.sqrt(Gx**2 + Gy**2)
    theta = np.arctan2(Gy, Gx) * 180 / np.pi
    theta[theta < 0] += 180

    # Suppression des non-maximums
    nms = np.zeros_like(G)
    h, w = G.shape

    for i in range(1, h-1):
        for j in range(1, w-1):
            angle = theta[i, j]

            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                q, r = G[i, j+1], G[i, j-1]
            elif 22.5 <= angle < 67.5:
                q, r = G[i-1, j+1], G[i+1, j-1]
            elif 67.5 <= angle < 112.5:
                q, r = G[i-1, j], G[i+1, j]
            else:
                q, r = G[i-1, j-1], G[i+1, j+1]

            if G[i, j] >= q and G[i, j] >= r:
                nms[i, j] = G[i, j]

    # Seuillage par hystérésis
    edges = np.zeros_like(nms)
    strong, weak = 255, 75

    strong_i, strong_j = np.where(nms >= sh)
    weak_i, weak_j = np.where((nms >= sb) & (nms < sh))

    edges[strong_i, strong_j] = strong
    edges[weak_i, weak_j] = weak

    for i in range(1, h-1):
        for j in range(1, w-1):
            if edges[i, j] == weak:
                if np.any(edges[i-1:i+2, j-1:j+2] == strong):
                    edges[i, j] = strong
                else:
                    edges[i, j] = 0

    return edges

def robert_edge_detection(im, sb=30, sh=70):
    Gx = np.array([[1, 0],
                   [0, -1]])
    Gy = np.array([[0, 1],
                   [-1, 0]])

    gx = cv2.filter2D(im.astype(np.float32), -1, Gx)
    gy = cv2.filter2D(im.astype(np.float32), -1, Gy)

    grad = np.sqrt(gx**2 + gy**2)
    grad = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX)
    grad = grad.astype(np.uint8)

    _, edges = cv2.threshold(grad, sb, 255, cv2.THRESH_BINARY)
    return edges


def prewitt_edge_detection(im, sb=30, sh=70):
    Gx = np.array([[-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]])
    Gy = np.array([[1, 1, 1],
                   [0, 0, 0],
                   [-1, -1, -1]])

    gx = cv2.filter2D(im.astype(np.float32), -1, Gx)
    gy = cv2.filter2D(im.astype(np.float32), -1, Gy)

    grad = np.sqrt(gx**2 + gy**2)
    grad = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX)
    grad = grad.astype(np.uint8)

    _, edges = cv2.threshold(grad, sb, 255, cv2.THRESH_BINARY)
    return edges


im = cv2.imread("../Images/cameraman.bmp", 0)
im = cv2.GaussianBlur(im, (5,5), 1)

sobel = sobel_edge_detection(im)
robert = robert_edge_detection(im)
prewitt = prewitt_edge_detection(im)

cv2.imshow("Sobel", sobel)
cv2.imshow("Robert", robert)
cv2.imshow("Prewitt", prewitt)
cv2.waitKey(0)
cv2.destroyAllWindows()
