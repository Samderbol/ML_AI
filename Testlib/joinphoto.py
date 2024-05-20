import cv2
import numpy as np

def stitch(image1, image2, orientation, zoom_factor):
    # 创建一个足够大的画布
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    # 缩小图像以加快特征匹配速度
    factor = zoom_factor
    image1_resized = cv2.resize(image1, (int(w1 * factor), int(h1 * factor)))
    image2_resized = cv2.resize(image2, (int(w2 * factor), int(h2 * factor)))

    # 使用SIFT检测器
    sift = cv2.SIFT_create()

    # 检测关键点和描述符
    keypoints1, descriptors1 = sift.detectAndCompute(image1_resized, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2_resized, None)

    # 使用FLANN匹配器进行特征匹配
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # 筛选好的匹配点
    good_matches = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good_matches.append(m)

    # 提取匹配点的坐标
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 缩放特征点坐标
    src_pts /= factor
    dst_pts /= factor

    # 使用RANSAC算法计算单应性矩阵
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)

    # 计算画布的大小
    if orientation == 'horizontal':
        canvas_width = w1 + w2
        canvas_height = max(h1, h2)
    elif orientation == 'vertical':
        canvas_width = max(w1, w2)
        canvas_height = h1 + h2
    else:
        raise ValueError('Invalid orientation')

        # 将第一张图像放置在画布上
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    canvas[:h1, :w1] = image1

    # 对第二张图像进行变换并放置在画布上
    transformed_image2 = cv2.warpPerspective(image2, M, (canvas_width, canvas_height), flags=cv2.WARP_INVERSE_MAP)
    transformed_image2[:h1, :w1] = 0
    canvas = cv2.addWeighted(canvas, 1, transformed_image2, 1, 0)

    return canvas


def crop_black_borders(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    return image[y:y + h, x:x + w]


if __name__ == '__main__':
    image1 = cv2.imread('1.jpg')
    image2 = cv2.imread('2.jpg')
    result = stitch(image1, image2, 'horizontal', 0.5)
    result = crop_black_borders(result)

    cv2.imwrite(f'12.jpg', result)