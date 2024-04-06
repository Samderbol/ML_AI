import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 1. 数据集准备
data = pd.read_excel("客户信息.xlsx")

# 设置中文字体为 macOS 系统自带的中文字体
plt.rcParams['font.family'] = 'Arial Unicode MS'

# 设置正常显示符号的字体
plt.rcParams['axes.unicode_minus'] = False

# 2. 数据可视化
plt.scatter(data['年龄(岁)'], data['收入(万元)'])
plt.xlabel('年龄(岁)')
plt.ylabel('收入(万元)')
plt.title('客户信息数据可视化')
plt.show()

# 3. 聚类
kmeans = KMeans(n_clusters=3)  # 假设分为3类
kmeans.fit(data)
data['Cluster'] = kmeans.labels_

# 4. 结果分析
centroids = kmeans.cluster_centers_
print("聚类中心点：")
print(centroids)

# 可视化聚类结果
colors = ['r', 'g', 'b']
for i in range(len(centroids)):
    plt.scatter(data[data['Cluster'] == i]['年龄(岁)'], data[data['Cluster'] == i]['收入(万元)'], c=colors[i], label=f'Cluster {i+1}')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, c='black', label='Centroids')
plt.xlabel('年龄(岁)')
plt.ylabel('收入(万元)')
plt.title('客户信息数据聚类结果')
plt.legend()
plt.show()
