import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path="G:\pycharm_text\data_base\ex1data1.txt"
data=pd.read_csv(path,header=None,names=['Population','Profit']) #给第一列命名population 第二列命名profit
def computeCost(X,y,theta):
    inner=np.power((X*theta.T)-y,2)  #误差hθ(x(i)-y(i))^2  np.power 计算矩阵中每个元素的次幂
    return np.sum(inner)/(2*len(X))  #返回代价公式 1/m sum error
data.insert(0,'ones',1)              #θ0 常数项
cols=data.shape[1]                   #列数
X=data.iloc[:,0:cols-1]              #第一列到倒数第二列
y=data.iloc[:,cols-1:cols]           #最后一列
X=np.matrix(X.values)
y=np.matrix(y.values)


def normalEqn(X, y):
    theta = np.linalg.inv(X.T@X)@X.T@y  # inv用来计算矩阵的逆   X.T@X等价于X.T.dot(X)
    return theta

final_theta2=normalEqn(X, y)
final_theta2=np.matrix(np.ravel(final_theta2))
print(computeCost(X,y,final_theta2))
print(final_theta2)
fig,ax = plt.subplots()
x=np.linspace(data.Population.min(),data.Population.max(),100)
y=final_theta2[0,0]+(final_theta2[0,1]*x)
ax.plot(x,y)
ax.scatter(data.Population,data.Profit)
plt.show()