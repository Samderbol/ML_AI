import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import color, feature, filters, io

# 读取图像
img = cv2.imread('room.tif')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 使用Sobel算子进行边缘检测
sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)

# 使用Laplacian算子进行边缘检测
laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)

# 对边缘检测结果进行二值化处理
_, sobelx_binary = cv2.threshold(np.abs(sobelx), 0.1, 1, cv2.THRESH_BINARY)
_, sobely_binary = cv2.threshold(np.abs(sobely), 0.1, 1, cv2.THRESH_BINARY)
_, laplacian_binary = cv2.threshold(np.abs(laplacian), 0.1, 1, cv2.THRESH_BINARY)

# 显示处理结果
plt.figure(figsize=(10, 10))
plt.subplot(3, 2, 1), plt.imshow(sobelx, cmap='gray'), plt.title('Sobel X')
plt.subplot(3, 2, 2), plt.imshow(sobelx_binary, cmap='gray'), plt.title('Sobel X Binary')
plt.subplot(3, 2, 3), plt.imshow(sobely, cmap='gray'), plt.title('Sobel Y')
plt.subplot(3, 2, 4), plt.imshow(sobely_binary, cmap='gray'), plt.title('Sobel Y Binary')
plt.subplot(3, 2, 5), plt.imshow(laplacian, cmap='gray'), plt.title('Laplacian')
plt.subplot(3, 2, 6), plt.imshow(laplacian_binary, cmap='gray'), plt.title('Laplacian Binary')
plt.show()

# 进行霍夫变换直线检测实验
edges = cv2.Canny(gray_img, 50, 150)
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        plt.plot([x1, x2], [y1, y2], 'r-', linewidth=2)
plt.title('Hough Lines')
plt.show()

# 全局阈值分割实验
_, thresholded_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

plt.imshow(thresholded_img, cmap='gray')
plt.title('Thresholded Image')
plt.show()
