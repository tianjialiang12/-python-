import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path="G:\pycharm_text\data_base\ex2data1.txt"
data=pd.read_csv(path,header=None,names=['Exam 1','Exam 2','Admitted'])
#print(data.head())
postive=data[data['Admitted'].isin([1])]  #将data中admitted为1的 称作postive
negtive=data[data['Admitted'].isin([0])]
fig,ax=plt.subplots(figsize=(12,8))
ax.scatter(postive['Exam 1'],postive['Exam 2'],s=50,c='b',marker='o',label='postive')
ax.scatter(negtive['Exam 1'],negtive['Exam 2'],s=50,c='r',marker='x',label='negtive')
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')


def sigmod(z):
    term=1/(1+np.exp(-z))
    return term
num=np.arange(-100,100,1)
fig,ax=plt.subplots(figsize=(12,8))
ax.plot(num,sigmod(num))
#plt.show()

def cost(theta,X,y):
    theta=np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first=np.multiply(-y,np.log(sigmod(X*theta.T)))
    secend=np.multiply((1-y),np.log(1-sigmod(X*theta.T)))
    return np.sum(first-secend)/len(X)

data.insert(0,'ones',1)
cols=data.shape[1]
X=data.iloc[:,0:cols-1]
y=data.iloc[:,cols-1:cols]
X = np.array(X.values)
y = np.array(y.values)
theta=np.matrix(np.zeros(3))
print(X.shape, theta.shape, y.shape)
#print(cost(theta,X,y))

def gradient(theta,X,y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    parameters=int(theta.shape[1])
    temp=np.matrix(np.zeros(theta.shape[1]))
    error=sigmod(X*theta.T)-y
    for i in range(parameters):
        term=np.multiply(error,X[:,i])
        temp[0,i]=np.sum(term)/len(X)
    return temp
#print(gradient(theta,X,y))
import scipy.optimize as opt
result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y))# func是要最小化的函数
                                                                        # x0是最小化函数的自变量
                                                                        # fprime是最小化的方法
                                                                        # args元组，是传递给优化函数的参数
# def grdientdescent(X,y,theta,alpha,iters):  #试试上面的函数和自己写的哪个好用
#     theta = np.matrix(theta)
#     X = np.matrix(X)
#     y = np.matrix(y)
#     temp=np.matrix(np.zeros(X.shape[1]))
#     parameters=X.shape[1]
#     for i in range(iters):
#         error=sigmod(X*theta.T)-y
#         for j in range(parameters):
#             term=np.multiply(error,X[:,j])
#             temp[0,j]=theta[0,j]-alpha*(1/len(X))*np.sum(term)
#         theta=temp
#     return theta
# result2=grdientdescent(X,y,theta,0.001,1000)
# print(result,result2)
# print(cost(result[0], X, y))
# print(cost(result2, X, y))

def predict(theta, X):
    probability = sigmod(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]
theta_min = np.matrix(result[0])
predictions = predict(theta_min, np.matrix(X))
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
accuracy = (sum(map(int, correct)) % len(correct))
print ('accuracy = {0}%'.format(accuracy))

