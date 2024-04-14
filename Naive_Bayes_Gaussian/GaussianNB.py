import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, confusion_matrix
import seaborn as sns

# 1. 数据集准备
data = pd.read_excel("肿瘤数据.xlsx")

# 划分特征和目标变量
X = data.iloc[:, :-1]  # 特征
y = data.iloc[:, -1]  # 目标变量

# 初始化性能指标列表
accuracy_list = []
precision_list = []
recall_list = []
f1_list = []

# 初始化ROC曲线参数列表
mean_fpr = np.linspace(0, 1, 100)
tprs = []

# 初始化混淆矩阵列表
conf_matrix_list = []

# 设置实验次数
num_experiments = 10

# 多次随机划分训练集和测试集进行实验
for i in range(num_experiments):
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

    # 2. 模型训练
    nb_classifier = GaussianNB()
    nb_classifier.fit(X_train, y_train)

    # 3. 模型预测
    y_pred = nb_classifier.predict(X_test)

    # 4. 计算指标
    accuracy_list.append(accuracy_score(y_test, y_pred))
    precision_list.append(precision_score(y_test, y_pred))
    recall_list.append(recall_score(y_test, y_pred))
    f1_list.append(f1_score(y_test, y_pred))

    # 6. 计算ROC曲线参数
    y_proba = nb_classifier.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0

    # 7. 计算混淆矩阵
    conf_matrix_list.append(confusion_matrix(y_test, y_pred))

# 计算指标的平均值和标准差
accuracy_mean = np.mean(accuracy_list)
accuracy_std = np.std(accuracy_list)
precision_mean = np.mean(precision_list)
precision_std = np.std(precision_list)
recall_mean = np.mean(recall_list)
recall_std = np.std(recall_list)
f1_mean = np.mean(f1_list)
f1_std = np.std(f1_list)

# 计算平均ROC曲线
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0

# 绘制平均ROC曲线
plt.plot(mean_fpr, mean_tpr, color='orange', label='Mean ROC')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Mean ROC Curve')
plt.legend()
plt.show()

# 计算混淆矩阵的平均值并四舍五入为整数
mean_conf_matrix = np.round(np.mean(conf_matrix_list, axis=0))

# 绘制混淆矩阵的热力图
plt.figure(figsize=(6, 4))
sns.heatmap(mean_conf_matrix, annot=True, fmt="g", cmap="Blues")
plt.title("Mean Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# 输出性能指标的平均值和标准差
print("平均准确率： {:.2f}，标准差： {:.2f}".format(accuracy_mean, accuracy_std))
print("平均精确率： {:.2f}，标准差： {:.2f}".format(precision_mean, precision_std))
print("平均召回率： {:.2f}，标准差： {:.2f}".format(recall_mean, recall_std))
print("平均F1分数： {:.2f}，标准差： {:.2f}".format(f1_mean, f1_std))
