import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 设置中文字体为 macOS 系统自带的中文字体
plt.rcParams['font.family'] = 'Arial Unicode MS'

# 设置正常显示符号的字体
plt.rcParams['axes.unicode_minus'] = False


# 步骤1：读取数据集并去重
data = pd.read_excel("新闻.xlsx")
data_unique = data.drop_duplicates(subset=['标题'])

# 步骤2：特征提取
# 分词
data_unique['标题分词'] = data_unique['标题'].apply(lambda x: " ".join(jieba.cut(x)))

# 文本向量化
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data_unique['标题分词'])

# 步骤3：聚类
k = 10  # 聚成10类
kmeans = KMeans(n_clusters=k)
kmeans.fit(X)
data_unique['类别'] = kmeans.labels_

# 合并结果到原始数据
data_merged = pd.merge(data, data_unique[['标题', '类别']], on='标题', how='left')

# 步骤4：可视化聚类结果
# 使用t-SNE进行降维
tsne = TSNE(n_components=2, random_state=0)
X_tsne = tsne.fit_transform(X.toarray())

# 绘制散点图
plt.figure(figsize=(10, 8))
for i in range(k):
    plt.scatter(X_tsne[data_unique['类别'] == i, 0], X_tsne[data_unique['类别'] == i, 1], label='类别 {}'.format(i))

plt.title('t-SNE 可视化聚类结果')
plt.xlabel('t-SNE 维度 1')
plt.ylabel('t-SNE 维度 2')
plt.legend()
plt.show()
