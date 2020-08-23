import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path="G:\pycharm_text\data_base\ex1data1.txt"
data=pd.read_csv(path,header=None,names=['Population','Profit']) #给第一列命名population 第二列命名profit
#print(data.head())
#print(data.describe())
data.plot(kind='scatter',x='Population',y='Profit',figsize=(12,8))
#plt.show()
def computeCost(X,y,theta):
    inner=np.power((X*theta.T)-y,2)  #误差hθ(x(i)-y(i))^2  np.power 计算矩阵中每个元素的次幂
    return np.sum(inner)/(2*len(X))  #返回代价公式 1/m sum error
data.insert(0,'ones',1)              #θ0 常数项
#print(data)
cols=data.shape[1]                   #列数
#print(cols)
X=data.iloc[:,0:cols-1]              #第一列到倒数第二列
y=data.iloc[:,cols-1:cols]           #最后一列
#print(X.head())
#print(y.head())
X=np.matrix(X.values)
y=np.matrix(y.values)
theta=np.matrix(np.array([0,0]))
#print(theta)
print(computeCost(X,y,theta))

def gradientDescent(X,y,theta,alpha,iters):
    temp=np.matrix(np.zeros(theta.shape))
    parameters=int(theta.ravel().shape[1])
    cost=np.zeros(iters)

    for i in range(iters):
        error=(X*theta.T)-y
        for j in range(parameters):
            term=np.multiply(error,X[:,j])
            temp[0,j]=theta[0,j]-((alpha/len(X))*np.sum(term))
        theta=temp
        cost[i]=computeCost(X,y,theta)
    return theta,cost

alpha = 0.01
iters = 1000

g , cost = gradientDescent(X,y,theta,alpha,iters)
#print(g)
x = np.linspace(data.Population.min(),data.Population.max(),100)
f = g[0,0]+(g[0,1]*x)
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
#plt.show()

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()

from sklearn import linear_model
model = linear_model.LinearRegression()
model.fit(X, y)
x = np.array(X[:, 1].A1)              #A1 将矩阵压平 等价于ravel()
f = model.predict(X).flatten()        # 压平 等价于ravel()
fig, ax = plt.subplots(figsize=(12,8))#新建一幅画布 规定尺寸 返回fig图片对象 ax轴对象
ax.plot(x, f, 'r', label='Prediction')#x y 颜色 标签
ax.scatter(data.Population, data.Profit, label='Traning Data')#x y 标签
ax.legend(loc=2)                      #图例
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()