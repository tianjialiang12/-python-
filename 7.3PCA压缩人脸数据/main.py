import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat
def pca(X):
    X = (X - X.mean()) / X.std()  #归一化
    X = np.matrix(X)
    cov = (X.T * X) / X.shape[0]  #计算协方差矩阵
    U, S, V = np.linalg.svd(cov)  #奇异值分解  cov=U @ S @ V.T
    return U, S, V   #U是一个m*m的矩阵 被称为主成分或者左奇异向量 S是m*n除对角线之外元素（奇异值）都为0  V是n*n矩阵 称为右奇异向量
def project_data(X, U, k):#原数据  主成分  降维后的列维度
    U_reduced = U[:,:k]   #将原矩阵映射到压缩后的矩阵上
    return np.dot(X, U_reduced)   # Z = X * U_reduced
def recover_data(Z, U, k):  #反向PCA  恢复维度
    U_reduced = U[:,:k]
    return np.dot(Z, U_reduced.T) # X = Z * U_reduced.T

faces = loadmat("G:\pycharm_text\data_base\ex7\ex7faces.mat")
X = faces['X']
print(X.shape)
# def plot_n_image(X, n): #渲染前n张人脸 n需要是一个平方数
#     pic_size = int(np.sqrt(X.shape[1]))
#     grid_size = int(np.sqrt(n))
#     first_n_images = X[:n, :]
#     fig, ax_array = plt.subplots(nrows=grid_size, ncols=grid_size,
#                                     sharey=True, sharex=True, figsize=(8, 8))
#     for r in range(grid_size):
#         for c in range(grid_size):
#             ax_array[r, c].imshow(first_n_images[grid_size * r + c].reshape((pic_size, pic_size)))
#             plt.xticks(np.array([]))
#             plt.yticks(np.array([]))
# plot_n_image(X,4)
# plt.show()

face = np.reshape(X[5,:], (32, 32))  #渲染第五张人脸 1024个像素reshape成 32*32的灰度图像
plt.imshow(face)
plt.show()

U, S, V = pca(X)
Z = project_data(X, U, 100)   #运行PCA  提取100个特征

X_recovered = recover_data(Z, U, 100)  #恢复数据 相当于压缩  但看起来并太大的不同  特征由1024》》100
face = np.reshape(X_recovered[3,:], (32, 32))
plt.imshow(face)
plt.show()