import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat

data = loadmat("G:\pycharm_text\data_base\ex8\ex8data1.mat")
X = data['X']
print(X.shape)
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(X[:,0], X[:,1])
plt.show()
def estimate_gaussian(X):  #返回每个特征（列）的均值和方差
    mu = X.mean(axis=0)
    sigma = X.var(axis=0)
    return mu, sigma
mu, sigma = estimate_gaussian(X)
# print(mu, sigma)

Xval = data['Xval']
yval = data['yval']
from scipy import stats
dist = stats.norm(mu[0], sigma[0]) #得到均值与方差对应的正态分布函数 mu[0] sigma[0]意思是对第一个特征（即正态分布的x轴坐标）进行拟合
# print(dist.pdf(X[:,0])[0:50])    #输出第一列（即x）前五十个数字的概率密度
p = np.zeros((X.shape[0], X.shape[1]))
p[:,0] = stats.norm(mu[0], sigma[0]).pdf(X[:,0])
p[:,1] = stats.norm(mu[1], sigma[1]).pdf(X[:,1])      #将X中所有数据对应的概率密度都放到P中
# print(p)
pval = np.zeros((Xval.shape[0], Xval.shape[1]))
pval[:,0] = stats.norm(mu[0], sigma[0]).pdf(Xval[:,0]) #将验证集中所有数据对应的概率密度都放到pval中
pval[:,1] = stats.norm(mu[1], sigma[1]).pdf(Xval[:,1])


def select_threshold(pval, yval): # 找出最优 阈值 ε
    best_epsilon = 0
    best_f1 = 0
    f1 = 0
    step = (pval.max() - pval.min()) / 1000
    for epsilon in np.arange(pval.min(), pval.max(), step):
        preds = pval < epsilon  #判断pval中的每个元素是否 < epsilon 并将判断结果（True，Flase）放入preds中 所以preds.shape=（n，2）
        tp = np.sum(np.logical_and(preds == 1, yval == 1)).astype(float) # logical_and 逻辑与
        #pval中某行的两个元素分别与 yval 中此行的元素做与  返回两个值 即pval第一个元素与yval的   pval第二个元素与yval的
        #np.sum即是返回 np.logical_and(preds == 1, yval == 1) 中所有True的个数  并转化为float类型
        fp = np.sum(np.logical_and(preds == 1, yval == 0)).astype(float)
        fn = np.sum(np.logical_and(preds == 0, yval == 1)).astype(float)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = (2 * precision * recall) / (precision + recall)
        if f1 > best_f1:
            best_f1 = f1
            best_epsilon = epsilon
    return best_epsilon, best_f1
epsilon, f1 = select_threshold(pval, yval)
outliers = np.where(p < epsilon) # P <ε 则y=1
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(X[:,0], X[:,1])
ax.scatter(X[outliers[0],0], X[outliers[0],1], s=50, color='r', marker='o')
plt.show()
# 利用验证集找出最优的ε并用于训练集查看结果