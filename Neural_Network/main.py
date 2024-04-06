import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# 设置中文字体为 macOS 系统自带的中文字体
plt.rcParams['font.family'] = 'Arial Unicode MS'

# 设置正常显示符号的字体
plt.rcParams['axes.unicode_minus'] = False

# 1. 数据集准备
data = pd.read_excel("产品评价.xlsx")

# 2. 中文分词
data['分词结果'] = data['评论'].apply(lambda x: ' '.join(jieba.cut(x)))

# 3. 特征提取
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['分词结果'])

# 4. 模型训练
X_train, X_test, y_train, y_test = train_test_split(X, data['评价'], test_size=0.2, random_state=42)
mlp_classifier = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)  # 定义神经网络模型
mlp_classifier.fit(X_train, y_train)

# 5. 模型预测
y_pred = mlp_classifier.predict(X_test)

# 6. 结果分析
accuracy = accuracy_score(y_test, y_pred)
print("模型准确率：", accuracy)

# 可视化结果分析
# 绘制混淆矩阵
from sklearn.metrics import confusion_matrix
import seaborn as sns

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('预测值')
plt.ylabel('真实值')
plt.title('混淆矩阵')
plt.show()
