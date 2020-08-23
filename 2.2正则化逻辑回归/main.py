import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
def sigmod(z):
    term=1/(1+np.exp(-z))
    return term
path ="G:\pycharm_text\data_base\ex2data2.txt"
data2 = pd.read_csv(path, header=None, names=['Test 1', 'Test 2', 'Accepted'])
#print(data2.head())
positive = data2[data2['Accepted'].isin([1])]
negative = data2[data2['Accepted'].isin([0])]
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(positive['Test 1'], positive['Test 2'], s=50, c='b', marker='o', label='Accepted')
ax.scatter(negative['Test 1'], negative['Test 2'], s=50, c='r', marker='x', label='Rejected')
ax.legend()
ax.set_xlabel('Test 1 Score')
ax.set_ylabel('Test 2 Score')
#plt.show()

degree = 5
x1 = data2['Test 1']
x2 = data2['Test 2']
data2.insert(3, 'Ones', 1)
for i in range(1, degree):
    for j in range(0, i):
        data2['F' + str(i) + str(j)] = np.power(x1, i-j) * np.power(x2, j) #特征太少，创建多项式特征
data2.drop('Test 1', axis=1, inplace=True)
data2.drop('Test 2', axis=1, inplace=True)
#print(data2.head())
def costReg(theta,X,y,leaningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first=np.multiply(-y,np.log(sigmod(X*theta.T)))
    second=np.multiply((1-y),np.log(1-sigmod(X*theta.T)))
    reg=leaningRate/(2*len(X))*np.sum(np.power(theta[:,1:theta.shape[1]],2))#
    term=np.sum(first-second)/len(X)+reg
    return term
def gradientReg(theta,X,y,learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    parameters=theta.shape[1]
    temp=np.zeros(theta.shape[1])
    error=sigmod(X*theta.T)-y
    for i in range(parameters):
        term = np.multiply(error, X[:,i])
        if(i==0):
            temp[i]=np.sum(term)/len(X)
        else:
            temp[i]= (np.sum(term) / len(X)) + ((learningRate / len(X)) * theta[:,i])
    return temp
cols = data2.shape[1]
X2 = data2.iloc[:,1:cols]
y2 = data2.iloc[:,0:1]
X2 = np.array(X2.values)
y2 = np.array(y2.values)
theta2 = np.zeros(11)
learningRate = 1
print(costReg(theta2,X2 ,y2 ,learningRate))
print(gradientReg(theta2,X2,y2,learningRate))
result2 = opt.fmin_tnc(func=costReg, x0=theta2, fprime=gradientReg, args=(X2, y2, learningRate))
print(result2)

def predict(theta, X):
    probability = sigmod(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]
theta_min = np.matrix(result2[0])
predictions = predict(theta_min, X2)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y2)]
accuracy = (sum(map(int, correct)) % len(correct))
print ('accuracy = {0}%'.format(accuracy))

from sklearn import linear_model#调用sklearn的线性回归包
model = linear_model.LogisticRegression(penalty='l2', C=1.0)
model.fit(X2, y2.ravel())
print(model.score(X2, y2))

























