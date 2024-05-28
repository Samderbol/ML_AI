import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_n twork import MLPClassifier
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

# 4. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, data['评价'], test_size=0.2, random_state=42)

# 5. 创建画布和子图
fig, ax = plt.subplots()
plt.ion()  # 打开交互模式，实时更新图表

# 6. 训练模型并记录准确率
mlp_classifier = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)  # 定义神经网络模型
accuracy_list = []

for i in range(100):  # 迭代100次
    mlp_classifier.partial_fit(X_train, y_train, classes=[0, 1])  # 部分拟合模型
    y_pred = mlp_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_list.append(accuracy)

    # 实时更新准确率曲线
    ax.clear()
    ax.plot(range(i + 1), accuracy_list, marker='o', linestyle='-')
    ax.set_xlabel('迭代次数')
    ax.set_ylabel('准确率')
    ax.set_title('准确率随迭代次数变化')
    plt.pause(0.1)

# 显示最终准确率曲线
plt.ioff()  # 关闭交互模式
plt.plot(range(100), accuracy_list, marker='o', linestyle='-')
plt.xlabel('迭代次数')
plt.ylabel('准确率')
plt.title('准确率随迭代次数变化')
plt.show()
