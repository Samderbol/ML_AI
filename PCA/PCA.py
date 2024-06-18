import os
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 下载图片
url = 'https://cs.nyu.edu/~roweis/data/olivettifaces.gif'
response = requests.get(url)
img = Image.open(BytesIO(response.content))

# 设定每张脸的宽和高
face_width, face_height = 92, 112

# 创建保存分割后图片的文件夹
if not os.path.exists('olivetti_faces'):
    os.makedirs('olivetti_faces')

# 拆分图片并保存
faces = []
for i in range(20):
    for j in range(20):
        face = img.crop((j * face_width, i * face_height, (j+1) * face_width, (i+1) * face_height))
        face = face.convert("L").resize((32, 32))  # 转为灰度图像并调整大小
        face.save(f'olivetti_faces/face_{i*20 + j + 1}.png')
        faces.append(np.array(face).reshape(1, 1024))

faces = np.array(faces).reshape(400, 1024)

# PCA降维
pca = PCA(n_components=3)
faces_pca = pca.fit_transform(faces)

# 显示前3个主成分
for i in range(3):
    plt.imshow(pca.components_[i].reshape(32, 32), cmap='gray')
    plt.title(f'主成分 {i+1}')
    plt.show()

# 打印主成分
print("主成分：", pca.components_)
