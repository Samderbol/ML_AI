import cv2
import matplotlib.pyplot as plt
print("OpenCV 版本:", cv2.__version__)
# 1. Read the digital image
image = cv2.imread('images.jpg', cv2.IMREAD_GRAYSCALE)

# 2. Display the original image histogram
plt.figure()
plt.hist(image.ravel(), 256, [0, 256])
plt.title('Original Image Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.ylim([0, 6000])
plt.xlim([0, 300])
plt.show()

# 3. Perform histogram equalization
equalized_image = cv2.equalizeHist(image)

# 4. Display the equalized image histogram
plt.figure()
plt.hist(equalized_image.ravel(), 256, [0, 256])
plt.title('Equalized Image Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.ylim([0, 6000])
plt.xlim([0, 300])
plt.show()

# 5. Display the original and equalized images
cv2.imshow('Original Image', image)
cv2.imshow('Equalized Image', equalized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()





