import pandas as pd
import numpy as np


# 定义梯度下降函数
def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    cost_history = []

    for i in range(num_iters):
        # 计算假设函数 h(x)
        h = X.dot(theta)

        # 计算损失函数
        loss = h - y

        # 计算梯度
        gradient = X.T.dot(loss) / m

        # 更新参数
        theta = theta - alpha * gradient

        # 计算损失函数值并保存历史值
        cost = np.sum(loss ** 2) / (2 * m)
        cost_history.append(cost)

        # 输出每次迭代的损失函数值
        if i % 100 == 0:
            print(f'Iteration {i}, Cost: {cost}')

    return theta, cost_history


# 生成示例数据
np.random.seed(0)
X = 2 * np.random.rand(100000, 1)
y = 4 + 3 * X + np.random.randn(100000, 1)

# 添加偏置项
X_b = np.c_[np.ones((100000, 1)), X]

# 初始化参数
theta = np.random.randn(2, 1)

# 设置超参数
alpha = 0.01
num_iters = 100000

# 执行梯度下降算法
theta_final, cost_history = gradient_descent(X_b, y, theta, alpha, num_iters)

# 打印最终参数值
print('Final Parameters:', theta_final)

# 打印最终损失函数值
print('Final Cost:', cost_history[-1])

