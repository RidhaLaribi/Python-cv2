import cv2
import numpy as np


# ==========================================
# PSNR
# ==========================================
def calculate_psnr(img1, img2):
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10((255.0 ** 2) / mse)


# ==========================================
# Custom Bilinear
# ==========================================
def custom_bilinear_resize(image, scale=2):
    h, w = image.shape[:2]
    H, W = int(h * scale), int(w * scale)

    x_out, y_out = np.meshgrid(np.arange(W), np.arange(H))
    src_x, src_y = x_out / scale, y_out / scale

    x0 = np.clip(np.floor(src_x).astype(int), 0, w - 2)
    y0 = np.clip(np.floor(src_y).astype(int), 0, h - 2)
    x1, y1 = x0 + 1, y0 + 1

    dx, dy = src_x - x0, src_y - y0

    if len(image.shape) == 3:
        dx = dx[..., None]
        dy = dy[..., None]

    Ia = image[y0, x0]
    Ib = image[y1, x0]
    Ic = image[y0, x1]
    Id = image[y1, x1]

    top = Ia * (1 - dx) + Ic * dx
    bottom = Ib * (1 - dx) + Id * dx
    return (top * (1 - dy) + bottom * dy).astype(np.uint8)


# ==========================================
# Adaptive (simple)
# ==========================================
def adaptive_edge_resize(image, scale=2):
    up = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return cv2.bilateralFilter(up, 5, 75, 75)


# ==========================================
# MAIN
# ==========================================
base = "Images_TP2_EXO6_SR/"
paths = {
    "HR": base + "HR_GT_512x512/",
    "LR256": base + "LR_256x256/",
    "LR128": base + "LR_128x128/"
}

results = {
    "2x_Nearest": [], "2x_Linear": [], "2x_Cubic": [], "2x_Custom": [],
    "4x_Nearest": [], "4x_Linear": [], "4x_Cubic": [], "4x_DoubleLinear": []
}

for i in range(1, 24):
    hr = cv2.imread(f"{paths['HR']}HR_GT_{i}.jpg")
    lr256 = cv2.imread(f"{paths['LR256']}LR_256_{i}.jpg")
    lr128 = cv2.imread(f"{paths['LR128']}LR_128_{i}.jpg")

    if hr is None or lr256 is None or lr128 is None:
        continue

    # -------- 2x --------
    rN = cv2.resize(lr256, (512, 512), cv2.INTER_NEAREST)
    rL = cv2.resize(lr256, (512, 512), cv2.INTER_LINEAR)
    rC = cv2.resize(lr256, (512, 512), cv2.INTER_CUBIC)
    rCustom = custom_bilinear_resize(lr256, 2)

    results["2x_Nearest"].append(calculate_psnr(hr, rN))
    results["2x_Linear"].append(calculate_psnr(hr, rL))
    results["2x_Cubic"].append(calculate_psnr(hr, rC))
    results["2x_Custom"].append(calculate_psnr(hr, rCustom))

    # -------- 4x --------
    r4N = cv2.resize(lr128, (512, 512), cv2.INTER_NEAREST)
    r4L = cv2.resize(lr128, (512, 512), cv2.INTER_LINEAR)
    r4C = cv2.resize(lr128, (512, 512), cv2.INTER_CUBIC)

    tmp = cv2.resize(lr128, (256, 256), cv2.INTER_LINEAR)
    r4D = cv2.resize(tmp, (512, 512), cv2.INTER_LINEAR)

    results["4x_Nearest"].append(calculate_psnr(hr, r4N))
    results["4x_Linear"].append(calculate_psnr(hr, r4L))
    results["4x_Cubic"].append(calculate_psnr(hr, r4C))
    results["4x_DoubleLinear"].append(calculate_psnr(hr, r4D))


# ==========================================
# PRINT TABLE
# ==========================================
def avg(x):
    return np.mean(x) if x else 0


print("\n================= PSNR RESULTS =================")
print(f"{'Method':<25} | {'Zoom':<5} | Avg PSNR")
print("-" * 50)

print(f"{'Nearest Neighbor':<25} | 2x   | {avg(results['2x_Nearest']):.2f}")
print(f"{'Bilinear (OpenCV)':<25} | 2x   | {avg(results['2x_Linear']):.2f}")
print(f"{'Bicubic (OpenCV)':<25} | 2x   | {avg(results['2x_Cubic']):.2f}")
print(f"{'Custom Bilinear':<25} | 2x   | {avg(results['2x_Custom']):.2f}")

print("-" * 50)

print(f"{'Nearest Neighbor':<25} | 4x   | {avg(results['4x_Nearest']):.2f}")
print(f"{'Bilinear (Direct)':<25} | 4x   | {avg(results['4x_Linear']):.2f}")

print(f"{'Bicubic (Direct)':<25} | 4x   | {avg(results['4x_Cubic']):.2f}")
print(f"{'Bilinear (2-step)':<25} | 4x   | {avg(results['4x_DoubleLinear']):.2f}")

print("================================================")
