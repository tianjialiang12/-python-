import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat
raw_data = loadmat("G:\pycharm_text\data_base\ex6\ex6data1.mat")
print(raw_data)
data = pd.DataFrame(raw_data['X'], columns=['X1', 'X2']) #添加两列 取名X1 X2
# print(data)
data['y'] = raw_data['y']   #添加一列
positive = data[data['y'].isin([1])]
negative = data[data['y'].isin([0])]
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(positive['X1'], positive['X2'], s=50, marker='x', label='Positive')
ax.scatter(negative['X1'], negative['X2'], s=50, marker='o', label='Negative')
ax.legend()
plt.show()
# print(data[['X1', 'X2']].shape,data['y'].shape)
# print(data['y'])

from sklearn import svm
svc = svm.LinearSVC(C=1, loss='hinge', max_iter=1000)#https://blog.csdn.net/weixin_41990278/article/details/93165330
svc.fit(data[['X1', 'X2']], data['y'])               #损失函数为hinge形式（不是平方）  最大迭代次数为1000次
svc.score(data[['X1', 'X2']], data['y'])
svc2 = svm.LinearSVC(C=100, loss='hinge', max_iter=1000)
svc2.fit(data[['X1', 'X2']], data['y'])
svc2.score(data[['X1', 'X2']], data['y'])

data['SVM 1 Confidence'] = svc.decision_function(data[['X1', 'X2']])  #返回每一个点到超平面的距离
# print(svc.decision_function(data[['X1', 'X2']]))
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(data['X1'], data['X2'], s=50, c=data['SVM 1 Confidence'], cmap='seismic')
ax.set_title('SVM (C=1) Decision Confidence')
plt.show()

data['SVM 2 Confidence'] = svc2.decision_function(data[['X1', 'X2']])
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(data['X1'], data['X2'], s=50, c=data['SVM 2 Confidence'], cmap='seismic')
ax.set_title('SVM (C=100) Decision Confidence')
plt.show()