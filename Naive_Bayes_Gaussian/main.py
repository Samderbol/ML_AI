# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import GaussianNB
# from sklearn.metrics import accuracy_score
#
# # 1. 数据集准备
# data = pd.read_excel("肿瘤数据.xlsx")
#
# # 划分特征和目标变量
# X = data.iloc[:, :-1]  # 特征
# y = data.iloc[:, -1]   # 目标变量
#
# # 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # 2. 模型训练
# nb_classifier = GaussianNB()
# nb_classifier.fit(X_train, y_train)
#
# # 3. 模型预测
# y_pred = nb_classifier.predict(X_test)
#
# # 4. 结果分析
# accuracy = accuracy_score(y_test, y_pred)
# print("模型准确率：", accuracy)
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import GaussianNB
# from sklearn.metrics import accuracy_score
#
# # 1. 数据集准备
# data = pd.read_excel("肿瘤数据.xlsx")
#
# # 划分特征和目标变量
# X = data.iloc[:, :-1]  # 特征
# y = data.iloc[:, -1]  # 目标变量
#
# # 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # 2. 模型训练
# nb_classifier = GaussianNB()
#
# # 记录训练集的损失和准确率
# train_loss = []
# train_accuracy = []
#
# for epoch in range(10):  # 假设训练10个epoch
#     nb_classifier.partial_fit(X_train, y_train, classes=np.unique(y))  # 部分拟合模型
#
#     # 3. 模型预测
#     y_pred_train = nb_classifier.predict(X_train)
#
#     # 计算训练集准确率
#     accuracy_train = accuracy_score(y_train, y_pred_train)
#     train_accuracy.append(accuracy_train)
#
#     # 4. 计算训练集损失（这里假设损失为零一）
#     loss_train = 1.0 - accuracy_train
#     train_loss.append(loss_train)
#
# # 4. 结果分析
# accuracy_test = accuracy_score(y_test, nb_classifier.predict(X_test))
# print("模型测试集准确率：", accuracy_test)
#
# # 绘制训练集损失和准确率图
# epochs = range(1, len(train_loss) + 1)
#
# plt.figure(figsize=(10, 5))
#
# plt.subplot(1, 2, 1)
# plt.plot(epochs, train_loss, 'b', label='Train loss')
# plt.title('Training loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
#
# plt.subplot(1, 2, 2)
# plt.plot(epochs, train_accuracy, 'r', label='Train accuracy')
# plt.title('Training accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
#
# plt.tight_layout()
# plt.show()
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

# 4. 计算指标
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# 5. 可视化指标
labels = ['Accuracy', 'Precision', 'Recall', 'F1-score']
values = [accuracy, precision, recall, f1]

plt.bar(labels, values, color=['blue', 'green', 'red', 'purple'])
plt.title('Performance Metrics')
plt.xlabel('Metrics')
plt.ylabel('Value')
plt.show()
#
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import GaussianNB
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, confusion_matrix
# import seaborn as sns
#
# # 1. 数据集准备
# data = pd.read_excel("肿瘤数据.xlsx")
#
# # 划分特征和目标变量
# X = data.iloc[:, :-1]  # 特征
# y = data.iloc[:, -1]   # 目标变量
#
# # 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # 2. 模型训练
# nb_classifier = GaussianNB()
# nb_classifier.fit(X_train, y_train)
#
# # 3. 模型预测
# y_pred = nb_classifier.predict(X_test)
#
# # 4. 计算指标
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)
#
# # 5. 可视化指标
# labels = ['Accuracy', 'Precision', 'Recall', 'F1-score']
# values = [accuracy, precision, recall, f1]
#
# plt.figure(figsize=(10, 5))
#
# plt.subplot(1, 2, 1)
# plt.bar(labels, values, color=['blue', 'green', 'red', 'purple'])
# plt.title('Performance Metrics')
# plt.xlabel('Metrics')
# plt.ylabel('Value')
#
# # 6. 绘制ROC曲线
# y_proba = nb_classifier.predict_proba(X_test)[:, 1]
# fpr, tpr, thresholds = roc_curve(y_test, y_proba)
#
# plt.subplot(1, 2, 2)
# plt.plot(fpr, tpr, color='orange', label='ROC')
# plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve')
# plt.legend()
#
# plt.tight_layout()
# plt.show()
#
# # 7. 绘制混淆矩阵
# conf_matrix = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(6, 4))
# sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
# plt.title("Confusion Matrix")
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.show()
