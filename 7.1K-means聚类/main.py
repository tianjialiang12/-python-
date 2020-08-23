import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat

#已知聚类中心 返回点在哪一个聚类中
def find_closest_centroids(X, centroids): #centroids:聚类中心的集合 列代表聚类中心的坐标（特征） 行代表几个聚类中心
    m=X.shape[0]                          #对于每一个数据  计算其到每一个聚类中心的距离  将最小的距离返回idx
    k=centroids.shape[0]                  #idx是一个1*m的数组 第一个放置m中第一个元素最近的聚类中心是几
    idx=np.zeros(m)
    for i in range(m):
        mindist=100000
        for j in range(k):
            dist = np.sum((X[i,:]-centroids[j,:])**2)
            if dist < mindist:
                mindist = dist
                idx[i] = j
    return idx
data = loadmat("G:\pycharm_text\data_base\ex7\ex7data2.mat")
X = data['X']
# print(X.shape)
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
idx = find_closest_centroids(X, initial_centroids)

# data2=pd.DataFrame(data['X'],columns=['X1','X2'])
# print(data2.head())
# plt.scatter(data2['X1'],data2['X2'])
# plt.show()

#已知每个点属于哪个聚类  返回每个聚类中所有元素的坐标平均值
def compute_centroids(X, idx, k):  #每一个聚类的所有元素 返回坐标的平均值
    m, n = X.shape
    centroids = np.zeros((k, n))
    for i in range(k):
        indices = np.where(idx == i)
        centroids[i, :] = (np.sum(X[indices, :], axis=1) / len(indices[0])).ravel()
    return centroids
# print(compute_centroids(X, idx, 3))

def run_k_means(X, initial_centroids, max_iters):
    m, n = X.shape
    k = initial_centroids.shape[0]
    idx = np.zeros(m)
    centroids = initial_centroids
    for i in range(max_iters):
        idx = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, idx, k)
    return idx, centroids
################################### K-means #########################
#1.先随机选取三个聚类中心
#2.将所有的元素按照距离分为三类
#3.计算三个聚类中所有元素的坐标的平均值，作为新的聚类中心
#4.重复2
#5.重复3  一直到迭代次数最大或者 坐标不再改变
idx, centroids = run_k_means(X, initial_centroids, 15)
cluster1 = X[np.where(idx == 0)]
cluster2 = X[np.where(idx == 1)]
cluster3 = X[np.where(idx == 2)]
plt.scatter(cluster1[:,0], cluster1[:,1], s=30, color='r', label='Cluster 1')
plt.scatter(cluster2[:,0], cluster2[:,1], s=30, color='g', label='Cluster 2')
plt.scatter(cluster3[:,0], cluster3[:,1], s=30, color='b', label='Cluster 3')
plt.legend()
plt.show()