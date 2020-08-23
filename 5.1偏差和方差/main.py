import numpy as np
import scipy.io as sio
import scipy.optimize as opt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
def load_data():
    d = sio.loadmat("G:\pycharm_text\data_base\ex5data1.mat")
    return map(np.ravel, [d['X'], d['y'], d['Xval'], d['yval'], d['Xtest'], d['ytest']])
    # map的用法 与下面一行等价  将后面的元素全部执行第一个位置的函数并返回arra类型的数据
    # return d['X'].ravel(),d['y'].ravel(),d['Xval'].ravel(),d['yval'].ravel(),d['Xtest'].ravel(),d['ytest'].ravel()
X, y, Xval, yval, Xtest, ytest = load_data()
# print(X.shape)
# plt.scatter(X,y,c='r')
# plt.show()
X, Xval, Xtest = [np.insert(x.reshape(x.shape[0], 1), 0, np.ones(x.shape[0]), axis=1) for x in (X, Xval, Xtest)]
#就是将三个数列转置后再加一列  与下面的代码效果相同
# X = X.reshape(X.shape[0],1)
# X = np.insert(X,0,np.ones(X.shape[0]),axis=1)
# Xval = Xval.reshape(Xval.shape[0],1)
# Xval = np.insert(Xval,0,np.ones(Xval.shape[0]),axis=1)
# Xtest = Xtest.reshape(Xtest.shape[0],1)
# Xtest = np.insert(Xtest,0,np.ones(Xtest.shape[0]),axis=1)
# print(X.shape,Xval.shape,Xtest.shape)
##################################################################################################################
def cost(theta, X, y):
    m = X.shape[0]
    error = X @ theta - y  # m*1
    square_sum = error.T @ error  #行向量 * 列向量 = 平方和
    cost = square_sum / (2 * m)   #往常是用np.sum  实现和
    return cost
def gradient(theta, X, y):
    m = X.shape[0]
    inner = X.T @ (X @ theta - y)  # 相当于x(i)*[X*theta - y]
    return inner / m               # (m,n).T @ (m, 1) -> (n, 1)
def regularized_gradient(theta, X, y, l=1):
    m = X.shape[0]
    regularized_term = theta.copy()  # same shape as theta
    regularized_term[0] = 0  # don't regularize intercept theta
    regularized_term = (l / m) * regularized_term
    return gradient(theta, X, y) + regularized_term
def regularized_cost(theta, X, y, l=1):
    m = X.shape[0]
    regularized_term = (l / (2 * m)) * np.power(theta[1:], 2).sum()
    return cost(theta, X, y) + regularized_term
def linear_regression_np(X, y, l=1):
    theta = np.ones(X.shape[1])
    res = opt.minimize(fun=regularized_cost,
                       x0=theta,
                       args=(X, y, l),
                       method='TNC',
                       jac=regularized_gradient,
                       options={'disp': True})
    return res
final_theta = linear_regression_np(X, y, l=0).get('x')

# b = final_theta[0] # intercept
# m = final_theta[1] # slope
# plt.scatter(X[:,1], y, label="Training data")
# plt.plot(X[:, 1], X[:, 1]*m + b, label="Prediction")
# plt.legend(loc=2)
# plt.show()
################################  画出图训练集与交叉测试集的图像###############
# m = X.shape[0]
# training_cost, cv_cost = [], []
# for i in range(1, m + 1):
#     res = linear_regression_np(X[:i, :], y[:i], l=0)
#     tc = regularized_cost(res.x, X[:i, :], y[:i], l=0)
#     cv = regularized_cost(res.x, Xval, yval, l=0)
#     training_cost.append(tc)
#     cv_cost.append(cv)
# plt.plot(np.arange(1, m+1), training_cost, label='training cost')
# plt.plot(np.arange(1, m+1), cv_cost, label='cv cost')
# plt.legend(loc=1)
# plt.show()

############### 欠拟合  增加多项式特征################
def normalize_feature(df):      #归一化特征
    return df.apply(lambda column: (column - column.mean()) / column.std())
#apply是将前面的对象(df)应用于后面的函数
#lambda是匿名函数 也就是不用那么多条条框框 参数是colum  返回 (column - column.mean()) / column.std()
def poly_features(x, power, as_ndarray=False): #创建x x^2 x^3......的特征
    data = {'f{}'.format(i): np.power(x, i) for i in range(1, power + 1)}
    df = pd.DataFrame(data)
    return df.values if as_ndarray else df #如果ndarray为true 则返回不带f1 f2...的结果（只返回值）
def prepare_poly_data(*args, power):
    def prepare(x):
        df = poly_features(x, power=power)     #创建特征
        ndarr = normalize_feature(df).values    #归一化 将f1 f2等名字去掉 只取值
        return np.insert(ndarr, 0, np.ones(ndarr.shape[0]), axis=1) #在第一列插入全为1的偏置项
    return [prepare(x) for x in args]

X, y, Xval, yval, Xtest, ytest = load_data()
# print(poly_features(X, power=3))
X_poly, Xval_poly, Xtest_poly= prepare_poly_data(X, Xval, Xtest, power=8) #创建8次的多项式特征 添加偏置项 共9列
# print(X_poly[:3, :],X_poly.shape)

def plot_learning_curve(X, y, Xval, yval, l=0):  #画训练集与交叉测试集的曲线
    training_cost, cv_cost = [], []
    m = X.shape[0]
    for i in range(1, m + 1):
        res = linear_regression_np(X[:i, :], y[:i], l=l)
        tc = cost(res.x, X[:i, :], y[:i])
        cv = cost(res.x, Xval, yval)
        training_cost.append(tc)
        cv_cost.append(cv)
    plt.plot(np.arange(1, m + 1), training_cost, label='training cost')
    plt.plot(np.arange(1, m + 1), cv_cost, label='cv cost')
    plt.legend(loc=1)
plot_learning_curve(X_poly, y, Xval_poly, yval, l=1)  #过拟合  需要增加l的值
plt.show()
plot_learning_curve(X_poly, y, Xval_poly, yval, l=100)#欠拟合  减少l的值
plt.show()

####################### 画出 J(Θ)-λ 的函数图像 判断λ的大概位置 ############################################
l_indicate=[0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
training_cost, cv_cost = [], []
for i in l_indicate:
    res=linear_regression_np(X_poly,y,i)
    tc=cost(res.x,X_poly,y)
    cv=cost(res.x,Xval_poly,yval)
    training_cost.append(tc)
    cv_cost.append(cv)
plt.plot(l_indicate, training_cost, label='training')
plt.plot(l_indicate, cv_cost, label='cross validation')
plt.legend(loc=2)
plt.xlabel('lambda')
plt.ylabel('cost')
plt.show()  #可以看出曲线拐点在 0.3 1 处均有

print(l_indicate[np.argmin(cv_cost)])  #看看拐点在哪里 得到λ的最优值  argmin就是当后面括号中的值取最小时 对应的索引
#但此时最小的代价对应的 λ 是交叉测试集的
#   使用测试集 将theta带入计算代价
for l in l_indicate:
    theta = linear_regression_np(X_poly, y, l).x
    print('test cost(l={}) = {}'.format(l, cost(theta, Xtest_poly, ytest)))
#训练集的目的就是得出theta 交叉测试集是用来配合训练集得出 λ 的最优值  测试集用来验证 与交叉验证集得到的λ不一致 以测试集为准
#最终λ为0.3