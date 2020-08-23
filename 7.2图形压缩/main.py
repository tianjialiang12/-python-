import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat
from IPython.display import Image
def find_closest_centroids(X, centroids): #centroids:聚类中心的集合 列代表聚类中心的坐标（特征） 行代表几个聚类中心
    m=X.shape[0]                          #对于每一个数据  计算其到每一个聚类中心的距离  将最小的距离返回idx
    k=centroids.shape[0]                  #idx是 一个1*m的数组 第一个放置m中第一个元素最近的聚类中心是几
    idx=np.zeros(m)                       #已知聚类中心 返回点在哪一个聚类中
    for i in range(m):
        mindist=100000
        for j in range(k):
            dist = np.sum((X[i,:]-centroids[j,:])**2)
            if dist < mindist:
                mindist = dist
                idx[i] = j
    return idx
def compute_centroids(X, idx, k):  #每一个聚类的所有元素 返回坐标的平均值
    m, n = X.shape
    centroids = np.zeros((k, n))
    for i in range(k):
        indices = np.where(idx == i)
        centroids[i, :] = (np.sum(X[indices, :], axis=1) / len(indices[0])).ravel()
    return centroids
def run_k_means(X, initial_centroids, max_iters):
    m, n = X.shape
    k = initial_centroids.shape[0]
    idx = np.zeros(m)
    centroids = initial_centroids
    for i in range(max_iters):
        idx = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, idx, k)
    return idx, centroids
Image(filename="G:\pycharm_text\data_base\ex7\smallbird.png")
image_data = loadmat("G:\pycharm_text\data_base\ex7\smallbird.mat")
A = image_data['A']
# print(A.shape) #返回值为（128，128，3） 表示array类型中有128个元素 每个元素有128行 3列
A = A / 255
X = A.reshape(A.shape[0] * A.shape[1], A.shape[2]) #转化为（128*128，3）
# print(X.shape)

def init_centroids(X, k):
    m, n = X.shape
    centroids = np.zeros((k, n))
    idx = np.random.randint(0, m, k)
    for i in range(k):
        centroids[i, :] = X[idx[i], :]
    return centroids
initial_centroids = init_centroids(X, 16)  #随机选取16个聚类中心
idx, centroids = run_k_means(X, initial_centroids, 10)#迭代十次
print(idx,idx.dtype,'\n')
print(idx.astype(int),idx.astype(int).dtype,'\n')
X_recovered = centroids[idx.astype(int),:]  #centroids中元素本来是float64的  idx中元素是[8,8,8,6,6,6...]等
                                            #相当于将所有点的坐标都分16类（取决于离哪个聚类中心更近）
X_recovered = X_recovered.reshape(A.shape[0], A.shape[1], A.shape[2])
plt.imshow(X_recovered)
plt.show()

####################### 用sklearn实现Kmeans ################
from skimage import io
pic = io.imread("G:\pycharm_text\data_base\ex7\smallbird.png") / 255.
io.imshow(pic)
plt.show()  #加载原图
# print(pic.shape)   #(128,128,3)
data = pic.reshape(128*128, 3)
from sklearn.cluster import KMeans#导入kmeans库
model = KMeans(n_clusters=16, n_init=100, n_jobs=-1) #https://www.cnblogs.com/wuchuanying/p/6218486.html
model.fit(data)

centroids = model.cluster_centers_
# print(centroids.shape)   #(16,3)
C = model.predict(data)  #每个点的预测值 即属于哪个聚类中心
# print(C.shape)           #(16384,)
# print(centroids[C].shape)#(16384,3)  相当于将所有点的坐标都分16类（取决于离哪个聚类中心更近） 与第54行作用相同
compressed_pic = centroids[C].reshape((128,128,3))

fig, ax = plt.subplots(1, 2)
ax[0].imshow(pic)
ax[1].imshow(compressed_pic)
plt.show()