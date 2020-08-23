import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat
from sklearn import svm

def gaussian_kernel(x1, x2, sigma):
    return np.exp(-(np.sum((x1 - x2) ** 2) / (2 * (sigma ** 2))))

raw_data = loadmat("G:\pycharm_text\data_base\ex6\ex6data2.mat")
data = pd.DataFrame(raw_data['X'], columns=['X1', 'X2'])  #值 列名
data['y'] = raw_data['y']
positive = data[data['y'].isin([1])]
negative = data[data['y'].isin([0])]
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(positive['X1'], positive['X2'], s=30, marker='x', label='Positive')
ax.scatter(negative['X1'], negative['X2'], s=30, marker='o', label='Negative')
ax.legend()
plt.show()

svc = svm.SVC(C=100, gamma=10, probability=True)  #https://blog.csdn.net/weixin_41990278/article/details/93137009
svc.fit(data[['X1', 'X2']], data['y'])
svc.score(data[['X1', 'X2']], data['y'])
data['Probability'] = svc.predict_proba(data[['X1', 'X2']])[:,0]
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(data['X1'], data['X2'], s=30, c=data['Probability'], cmap='Reds')
plt.show()



raw_data = loadmat("G:\pycharm_text\data_base\ex6\ex6data3.mat")
X = raw_data['X']                 #因为不需要作图 所以没有用上面那种pd.DataFram的形式加载数据
Xval = raw_data['Xval']           #svm.fit()的两个变量一个要求是(n,2) 一个要求是(1,n)
y = raw_data['y'].ravel()
yval = raw_data['yval'].ravel()
print(np.matrix(X).shape,np.matrix(y).shape)
C_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
gamma_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
best_score = 0
best_params = {'C': None, 'gamma': None}
for C in C_values:
    for gamma in gamma_values:
        svc = svm.SVC(C=C, gamma=gamma)
        svc.fit(X, y)
        score = svc.score(Xval, yval)
        if score > best_score:
            best_score = score
            best_params['C'] = C
            best_params['gamma'] = gamma
print(best_score, best_params)