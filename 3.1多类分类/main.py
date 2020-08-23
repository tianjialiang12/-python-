import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import matplotlib
import scipy.optimize as opt
from sklearn.metrics import classification_report#这个包是评价报告

def load_data(path,transpose=True):
    data=sio.loadmat(path)
    y = data.get('y')  #(5000,1)
    y = y.ravel()
    X=data.get('X')
    if transpose:
        X = np.array([im.reshape((20, 20)).T for im in X])#原始数据对图像进行了翻转  这里需要转置回来
        #每幅图有400个像素点  需要转置
        #取每一行（400个数）构建一个20*20的矩阵  得到一个array 有5000个元素 每个元素都是20*20的矩阵
        X = np.array([im.reshape(400) for im in X])       #将每个20*20的矩阵压平 得到一个5000*400的矩阵
    return X,y
X, y = load_data("G:\pycharm_text\data_base\ex3data1.mat")
#print(X.shape)
#print(y.shape)

#对上面数据处理的例子
# q=np.arange(1,28,1).reshape(3,9)
# print(q)
# q=np.array([im.reshape((3, 3)).T for im in q])
# print(q)
# q= np.array([im.reshape(9) for im in q])
# print(q)

def plot_an_image(image):
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.matshow(image.reshape((20, 20)), cmap=matplotlib.cm.binary)
    plt.xticks(())  # 去除坐标轴
    plt.yticks(())
pick_one = np.random.randint(0, 5000) #randint整型随机数 random浮点随机数
#plot_an_image(X[pick_one, :])         #random.uniform(1.1,5.4) 产生  1.1 到 5.4 之间的随机浮点数，区间可以不是整数
#plt.show()                            #random.choice('tomorrow') 从序列中随机选取一个元素
#print('this should be {}'.format(y[pick_one])) #format用来填充字符串

def plot_100_image(X):
    size = int(np.sqrt(X.shape[1]))  #np.sqrt([1,4,9])  >>([1,2,3])
    sample_idx = np.random.choice(np.arange(X.shape[0]), 100)  # 随机选取5000张图片中的 100张
    sample_images = X[sample_idx, :]
    fig, ax_array = plt.subplots(nrows=10, ncols=10, sharey=True, sharex=True, figsize=(8, 8))
    #将画布分割成10行 10列 共享y轴 共享x轴
    for r in range(10):
        for c in range(10):
            ax_array[r, c].matshow(sample_images[10 * r + c].reshape((size, size)),cmap=matplotlib.cm.binary)
            plt.xticks(())
            plt.yticks(())  #去除坐标轴
# plot_100_image(X)
# plt.show()

raw_X, raw_y = load_data("G:\pycharm_text\data_base\ex3data1.mat")
#print(raw_X.shape)
#print(raw_y.shape)
X = np.insert(raw_X, 0, values=np.ones(raw_X.shape[0]), axis=1)#如果axis为0  则整个返回值被压扁为一维
#print(X)

y_matrix = []
for k in range(1, 11):
    y_matrix.append((raw_y == k).astype(int))    # 将y中的数字用0（不等于k），1（等于k）代替  见配图 "向量化标签.png"
#print(np.array(y_matrix),'\n')                  #y是10.. 1.. 2.. 3.. 所以最后一列就是前几个为1（用10代表了0）
y_matrix = [y_matrix[-1]] + y_matrix[:-1]        # 把最后一列放到第一列  具体见图
y = np.array(y_matrix)                           # 就是one-hot编码 也可以调用sklearn中的函数
#print(y)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def cost(theta, X, y):
    return np.mean(-y * np.log(sigmoid(X @ theta)) - (1 - y) * np.log(1 - sigmoid(X @ theta)))
def regularized_cost(theta, X, y, l=5):
    theta_j1_to_n = theta[1:]
    regularized_term = (l / (2 * len(X))) * np.power(theta_j1_to_n, 2).sum()
    return cost(theta, X, y) + regularized_term
def regularized_gradient(theta, X, y, l=5):
    theta_j1_to_n = theta[1:]
    regularized_theta = (l / len(X)) * theta_j1_to_n
    regularized_term = np.concatenate([np.array([0]), regularized_theta]) #拼接 因为i为0时 没有正则化项
    return gradient(theta, X, y) + regularized_term
def gradient(theta, X, y):
    return (1 / len(X)) * X.T @ (sigmoid(X @ theta) - y)
def logistic_regression(X, y, l=1.2):
    theta = np.zeros(X.shape[1])
    res = opt.minimize(fun=regularized_cost,
                       x0=theta,
                       args=(X, y, l),
                       method='TNC',
                       jac=regularized_gradient,
                       options={'disp': True})
    final_theta = res.x
    return final_theta
def predict(x, theta):
    prob = sigmoid(x @ theta)
    return (prob >= 0.5).astype(int)    #astype(int)放在判断语句后面 意思是如果为真则输出1 否则输出0
theta0 = logistic_regression(X, y[0])       #只训练了 0 图像识别
y_pred = predict(X, theta0)
#print('Accuracy={}'.format(np.mean(y[0] == y_pred)))

k_theta = np.array([logistic_regression(X, y[k]) for k in range(10)])#训练K维模型 即0-9的图像识别
prob_matrix = sigmoid(X @ k_theta.T)
#因为逻辑回归只能一次进行二分类  所以将y分为10组 进行十次二分类 比如第一次就只分类是否为0
#而theta0就是一个（401，1）的向量  它表示了20*20像素以及一个截距项的所有特征值（在是否为0这个分类器中的）
#k_theta就是一个（10，401）的向量（因为np.array将每个子theta看作了一行）
#prob_matrix就是X*k_theta的转置再用sigmoid函数处理  即（5000，401）*（401，10） 所以其为（5000，10）
#prob_matrix的意义为  每一列代表一个数字的5000幅图片预测情况 如第300幅图是手写0  则prob_matrix第300行应该为 1 0 0 0 0 0 0 0 0 0
# 在sigmoid处理下 都接近0 或者接近1 后续计算精确度以0.5为阈值即可
y_pred = np.argmax(prob_matrix, axis=1) #返回每一行最大数的索引 即预测值接近1的那个数
y_answer = raw_y.copy()                 #真实值  本来应该是数字几
y_answer[y_answer==10] = 0              #将10换成0
print(classification_report(y_answer, y_pred))#打印精确度报告


































