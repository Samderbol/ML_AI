# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score
#
# # 1. 数据集准备
# data = pd.read_excel("葡萄酒.xlsx")
#
# # 划分特征和目标变量
# X = data.iloc[:, 1:-1]  # 特征
# y = data.iloc[:, -1]    # 目标变量
#
# # 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # 2. 模型训练
# knn_classifier = KNeighborsClassifier(n_neighbors=3)  # K近邻分类器，设定K=3
# knn_classifier.fit(X_train, y_train)
#
# # 3. 模型预测
# y_pred = knn_classifier.predict(X_test)
#
# # 4. 结果分析
# accuracy = accuracy_score(y_test, y_pred)
# print("模型准确率：", accuracy)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_curve, roc_auc_score, confusion_matrix

# 1. 数据集准备
data = pd.read_excel("wine.xlsx")

# 划分特征和目标变量
X = data.iloc[:, 1:-1]  # 特征
y = data.iloc[:, -1]    # 目标变量

# 定义超参数范围
param_grid = {'n_neighbors': [3, 5, 7, 9]}  # 近邻参数k的范围

# 初始化模型
knn_classifier = KNeighborsClassifier()

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. 模型训练与调参
stratified_kfold = StratifiedKFold(n_splits=10)
grid_search = GridSearchCV(knn_classifier, param_grid, cv=stratified_kfold, scoring='accuracy')
grid_search.fit(X_train, y_train)

# 输出最佳参数
print("最佳超参数:", grid_search.best_params_)

# 3. 模型评估与预测
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# 评价指标计算
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)

print("模型准确率:", accuracy)
print("模型F1-score:", f1)
print("模型AUC:", auc)

# 4. 结果可视化与分析
# ROC曲线
y_proba = best_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label='ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# 混淆矩阵热力图
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
