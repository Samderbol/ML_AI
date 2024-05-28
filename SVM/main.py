# from sklearn import datasets
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
#
# # 读取Iris数据集
# iris = datasets.load_iris()
# X = iris.data
# y = iris.target
#
# # 划分数据集为训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#
# # 初始化SVM分类器
# svm_classifier = SVC(kernel='linear')
#
# # 训练模型
# svm_classifier.fit(X_train, y_train)
#
# # 对测试集进行预测
# y_pred = svm_classifier.predict(X_test)
#
# # 评估预测结果
# accuracy = accuracy_score(y_test, y_pred)
# print(f"模型准确率: {accuracy}")
#
# # 打印混淆矩阵
# conf_matrix = confusion_matrix(y_test, y_pred)
# print("混淆矩阵:")
# print(conf_matrix)
#
# # 打印分类报告
# class_report = classification_report(y_test, y_pred, target_names=iris.target_names)
# print("分类报告:")
# print(class_report)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 读取Iris数据集
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 初始化SVM分类器
svm_classifier = SVC(kernel='linear')

# 训练模型
svm_classifier.fit(X_train, y_train)

# 对测试集进行预测
y_pred = svm_classifier.predict(X_test)

# 评估预测结果
accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确率: {accuracy}")

# 打印混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
print("混淆矩阵:")
print(conf_matrix)

# 打印分类报告
class_report = classification_report(y_test, y_pred, target_names=iris.target_names)
print("分类报告:")
print(class_report)

# 绘制混淆矩阵图表
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# 将分类报告转换为数据框并绘制图表
import pandas as pd
import numpy as np

# 创建分类报告数据框
report_data = []
for label, metrics in classification_report(y_test, y_pred, target_names=iris.target_names, output_dict=True).items():
    if label in iris.target_names:
        report_data.append([label, metrics['precision'], metrics['recall'], metrics['f1-score'], metrics['support']])

report_df = pd.DataFrame(report_data, columns=['Class', 'Precision', 'Recall', 'F1-Score', 'Support'])

# 绘制分类报告图表
fig, ax = plt.subplots(figsize=(10, 6))
report_df.plot(x='Class', y=['Precision', 'Recall', 'F1-Score'], kind='bar', ax=ax)
plt.title('Classification Report')
plt.xlabel('Class')
plt.ylabel('Scores')
plt.ylim(0, 1.1)
plt.xticks(rotation=0)
plt.show()
