#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
机器学习的基本原理
假定已知的样本数据的分布符合sin 函数的特征，但事先不知道sin函数
通过一元三次方程来拟合sin函数曲线，使得最终获得的
一元三次方程能够符合数据样本的特征
需要求解的一元三次方程 y = a + b x + c x^2 + d x^3，
事实上需要求解的是a, b, c, d 4个系数
'''
import numpy as np
import math

# 在-pi 和 pi 之间创建2000个点，作为二维坐标系中的 x坐标 
x = np.linspace(-math.pi, math.pi, 2000)
# 通过sin函数，计算出2000个点的 y坐标
y = np.sin(x)

# a, b, c, d 是待拟合的多项式（一元3次方程）的4个系数，待求解
# 这里作为计算的初始条件，先随机给几个值
a = np.random.randn()
b = np.random.randn()
c = np.random.randn()
d = np.random.randn()

# 每次调整a、b、c、d数据的步伐大小
learning_rate = 1e-6
for t in range(2000):
    # Forward pass: compute predicted y
    # y = a + b x + c x^2 + d x^3
    # 对于给定的2000个点中的没一个点x的至，求解 y的值
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    # 将计算出来的y的值，与sin函数的标准值（样本数据）比较，求方差
    # 这是计算误差的标准方法，square即平方，sum及统计里的sigma
    loss = np.square(y_pred - y).sum()

    # 这个只是打个日志看看，可以看到loss是逐渐减小的，即拟合的曲线
    # 逐渐符合样本值的特征
    if t % 100 == 99:
        print(t, loss)

    # Backprop to compute gradients of a, b, c, d with respect to loss
    # 机器学习里叫后向传播，通过梯度下降方法逐步逼近样本集特征
    # 需要高等数学的求导数
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()

    # Update weights
    # 更新一元3次方程的系数，进行下一次迭代
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d

# 打印出最终拟合的一元三次方程，作为最接近sin函数的方程
# 为啥用一元三次方程拟合？而不是别的方程？这个可以追溯到泰勒公式
# 泰勒公式可以用函数在某一点的各阶导数值做系数构建一个多项式来近似表达这个函数
print(f'Result: y = {a} + {b}*x + {c}*x**2 + {d}*x**3')
