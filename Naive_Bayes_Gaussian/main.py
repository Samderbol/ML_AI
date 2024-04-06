import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# 1. 数据集准备
data = pd.read_excel("肿瘤数据.xlsx")

# 划分特征和目标变量
X = data.iloc[:, :-1]  # 特征
y = data.iloc[:, -1]   # 目标变量

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. 模型训练
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

# 3. 模型预测
y_pred = nb_classifier.predict(X_test)

# 4. 结果分析
accuracy = accuracy_score(y_test, y_pred)
print("模型准确率：", accuracy)
