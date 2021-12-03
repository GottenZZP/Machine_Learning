# coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def getFile():
    # 获取文件路径
    path = "ex1data2.txt"

    # 读取文件，并给文件中的两列起了别名
    data = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])

    # 进行特征均值归一化
    data = featureScaling(data)

    # 在文件第一排插入一列1，并取名为Ones
    data.insert(0, 'Ones', 1)
    return data


# 处理文件内容
def handleFile(data):
    # 获取数据行数
    cols = data.shape[1]
    # 取出X
    X = data.iloc[:, 0:cols - 1]
    y = data.iloc[:, cols - 1:cols]
    # 将X，y转化为numpy数组，好为后续求值
    X = np.matrix(X.values)
    y = np.matrix(y.values)
    # 初始化theta
    theta = np.matrix(np.array([[0], [0], [0]]))

    return X, y, theta


def featureScaling(data):
    data = (data - data.mean()) / data.std()
    return data


# 计算代价函数
def computerCost(X, y, theta):
    m = len(X)
    predictions = X * theta
    sq_errors = np.power((predictions - y), 2)
    j = sum(sq_errors) / (2 * m)
    return j


# 梯度下降法
def gradientDescent(X, y, theta, alpha, iters):
    # 创建一个和theta相同参数的临时数组
    temp = np.matrix(np.zeros(theta.shape))
    # 创建一个保存每次迭代的代价函数的值
    cost = np.zeros(iters)
    # theta迭代次数参数
    parameters = int(theta.shape[0])
    m = len(X)
    # 迭代梯度下降
    for i in range(iters):
        # 计算预测值与实际值的的误差
        error = X * theta - y
        # 循环迭代每一个theta值
        for j in range(parameters):
            term = np.multiply(error, X[:, j])
            temp[j] = theta[j] - (alpha / m) * np.sum(term)
        theta = temp
        cost[i] = computerCost(X, y, theta)

    return theta, cost


def TwoDPlot(theta, data):
    x = np.linspace(data.Population.min(), data.Population.max(), 100)
    f = float(theta[0]) + (float(theta[1]) * x)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(x, f, 'r', label='Prediction')
    ax.scatter(data.Population, data.Profit, label='Traning Data')
    ax.legend(loc=4)
    ax.set_xlabel('Population')
    ax.set_ylabel('Profit')
    ax.set_title('Predicted Profit vs. Population Size')
    plt.show()


def ThreeDplot(theta, data, x1, x2, y):
    x = np.array([np.linspace(data.Size.min(), data.Size.max(), 100),
                  np.linspace(data.Bedrooms.min(), data.Bedrooms.max(), 100)])
    f = float(theta[0]) + float(theta[1]) * x[0] + float(theta[2]) * x[1]
    ax = plt.axes(projection='3d')
    ax.plot3D(x[0], x[1], f, 'gray')
    ax.scatter3D(x1, x2, y, c=y, cmap='Greens')


if __name__ == '__main__':
    data = getFile()
    X, y, theta = handleFile(data)
    theta, cost = gradientDescent(X, y, theta, 0.01, 1000)
    J = computerCost(X, y, theta)
    print(J)

    # TwoDPlot(theta, data)
    ThreeDplot(theta, data, X[:, 1], X[:, 2], y)
    # print(f)
    # plt.plot(np.arange(1000), cost, 'r')
    plt.show()
