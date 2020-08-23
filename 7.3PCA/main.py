import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat
data = loadmat("G:\pycharm_text\data_base\ex7\ex7data1.mat")
# print(data)
X=data['X']
plt.scatter(X[:,0],X[:,1])
# plt.show()
def pca(X):
    X = (X - X.mean()) / X.std()  #归一化
    X = np.matrix(X)
    cov = (X.T * X) / X.shape[0]  #计算协方差矩阵
    U, S, V = np.linalg.svd(cov)  #奇异值分解  cov=U @ S @ V.T
    return U, S, V   #U是一个m*m的矩阵 被称为主成分或者左奇异向量 S是m*n除对角线之外元素（奇异值）都为0  V是n*n矩阵 称为右奇异向量
U, S, V = pca(X)
def project_data(X, U, k):#原数据  主成分  降维后的列维度
    U_reduced = U[:,:k]   #将原矩阵映射到压缩后的矩阵上
    return np.dot(X, U_reduced) # Z = X * U_reduced
Z = project_data(X, U, 1)

def recover_data(Z, U, k):  #反向PCA  恢复维度
    U_reduced = U[:,:k]
    return np.dot(Z, U_reduced.T)
X_recovered = recover_data(Z, U, 1) # X = Z * U_reduced.T

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(list(X_recovered[:, 0]), list(X_recovered[:, 1]))
plt.show()