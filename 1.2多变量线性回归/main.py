import numpy as np
import pandas as pd
import matplotlib.pylab as plt

path="G:\pycharm_text\data_base\ex1data2.txt"
data=pd.read_csv(path,header=None,names=['size','bedrooms','price'])
#print(data.head())
data=(data-data.mean())/data.std()
#print(data.head())
data.insert(0,'ones',1)
#print(data.head())
cols=data.shape[1]
X=data.iloc[:,0:cols-1]
y=data.iloc[:,cols-1:cols]
print(y)
X=np.matrix(X.values)
y=np.matrix(y.values)
theta=np.matrix(np.zeros(3))
#print(theta)
def computecost(X,y,theta):
    inner=np.power((X*theta.T)-y,2)
    return np.sum(inner)/(2*len(X))
def grdientdescent(X,y,theta,alpha,iters):
    temp=np.matrix(np.zeros(X.shape[1]))
    parameters=X.shape[1]
    parameters = int(theta.ravel().shape[1])
    cost=np.zeros(iters)
    for i in range(iters):
        error=(X*theta.T)-y
        for j in range(parameters):
            term=np.multiply(error,X[:,j])
            temp[0,j]=theta[0,j]-alpha*(1/len(X))*np.sum(term)
        theta=temp
        cost[i]=computecost(X,y,theta)
    return theta,cost
g,cost=grdientdescent(X,y,theta,0.01,1000)
print(computecost(X,y,g))
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(1000), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()


