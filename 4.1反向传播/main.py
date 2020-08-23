import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import matplotlib
import scipy.optimize as opt
from sklearn.metrics import classification_report#这个包是评价报告
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
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

X, _ = load_data("G:\pycharm_text\data_base\ex4data1.mat")

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
X_raw, y_raw = load_data("G:\pycharm_text\data_base\ex4data1.mat", transpose=False)
X = np.insert(X_raw, 0, np.ones(X_raw.shape[0]), axis=1)#增加全部为1的一列
# print(y_raw,y_raw.shape)
def expand_y(y):
    res=[]
    j=0
    for i in y:                    #y_raw中是10...(500)  1...(500)等等
        y_array=np.zeros(10)       #返回值 [0000000001](500)
        y_array[i-1]=1             #      [100000000](500)
        res.append(y_array)        #      [0100000000](500)
    return np.array(res)           #......
y = expand_y(y_raw)                #      [0000000010](500)
# np.set_printoptions(threshold=np.inf)

### 也可以用sklearn的内置函数完成onehot编码  但其要求y_raw为(5000,1)的矩阵  因为在本次数据获取中将y ravel()了 所以不能用了 效果一样
# from sklearn.preprocessing import OneHotEncoder
# encoder = OneHotEncoder(sparse=False)
# y_onehot = encoder.fit_transform(y_raw)
# print(y_onehot.shape)
##################################################################################################################

def load_weight(path):
    data = sio.loadmat(path)
    return data['Theta1'], data['Theta2']
t1, t2 = load_weight("G:\pycharm_text\data_base\ex4weights.mat")
#print(t1.shape, t2.shape)
def serialize(a, b):   #为了在以后的函数中传参方便一点 不用多写几个theta
    return np.concatenate((np.ravel(a), np.ravel(b))) #theta1（25,401），theta2（10,26）
theta = serialize(t1, t2)                             # 扁平化参数，25*401+10*26=10285  即1*10285
def deserialize(seq):                                 #将theta再还原。。。。。
    return seq[:25 * 401].reshape(25, 401), seq[25 * 401:].reshape(10, 26)
def feed_forward(theta, X):
    t1, t2 = deserialize(theta)  # t1: (25,401) t2: (10,26)
    m = X.shape[0]
    a1 = X  # 5000 * 401
    z2 = a1 @ t1.T  # 5000 * 25
    a2 = np.insert(sigmoid(z2), 0, np.ones(m), axis=1)  # 5000*26
    z3 = a2 @ t2.T  # 5000 * 10
    h = sigmoid(z3)  # 5000*10, this is h_theta(X)
    return a1, z2, a2, z3, h  # 需要这些参数去反向传播
a1,z2,a2,z3,h=feed_forward(theta,X)  #X就是X_raw 加了一列常数
# print(y.shape,h.shape)
def cost(theta, X, y):
    m = X.shape[0]  # get the data size m
    _, _, _, _, h = feed_forward(theta, X)
    pair_computation = -np.multiply(y, np.log(h)) - np.multiply((1 - y), np.log(1 - h))
    return pair_computation.sum() / m
# print(cost(theta, X, y))
def regularized_cost(theta, X, y, l=1):
    t1, t2 = deserialize(theta)  # t1: (25,401) t2: (10,26)
    m = X.shape[0]                                         # 将所有的theta平方后都加一个系数作为惩罚项
    reg_t1 = (l / (2 * m)) * np.power(t1[:, 1:], 2).sum()  # 输入层到隐藏层的theta惩罚项  忽略了常数项
    reg_t2 = (l / (2 * m)) * np.power(t2[:, 1:], 2).sum()  # 隐藏层到输出层的theta惩罚项  忽略了常数项
    return cost(theta, X, y) + reg_t1 + reg_t2             # 具体公式可以参考4-1NNback....里面的 和笔记本的本质相同
# print(regularized_cost(theta, X, y))

#######################  反向传播  ############################
# print(X.shape,y.shape,t1.shape,t2.shape,theta.shape)
def sigmoid_gradient(z):
     return np.multiply(sigmoid(z), 1 - sigmoid(z))

def gradient(theta, X, y):         #和笔记本的计算过程完全一致
    t1, t2 = deserialize(theta)  # t1: (25,401) t2: (10,26)
    m = X.shape[0]
    delta1 = np.zeros(t1.shape)  # (25, 401)
    delta2 = np.zeros(t2.shape)  # (10, 26)
    a1, z2, a2, z3, h = feed_forward(theta, X)
    for i in range(m):  #循环5000次 一个一个数字来得到梯度  隐藏层有25个神经元（加一个偏置项）
        a1i = a1[i, :]  # (1, 401)
        z2i = z2[i, :]  # (1, 25)
        a2i = a2[i, :]  # (1, 26)
        hi = h[i, :]    # (1, 10)
        yi = y[i, :]    # (1, 10)
        d3i = hi - yi   # (1, 10)  #输出层的δ
        z2i = np.insert(z2i, 0, np.ones(1))  # make it (1, 26) to compute d2i
        d2i = np.multiply(t2.T @ d3i, sigmoid_gradient(z2i))  # (1, 26)   #隐藏层的δ
        delta2 += np.matrix(d3i).T @ np.matrix(a2i)  # (1, 10).T @ (1, 26) -> (10, 26)       #隐藏层到输出层的Δ
        delta1 += np.matrix(d2i[1:]).T @ np.matrix(a1i)  # (1, 25).T @ (1, 401) -> (25, 401) #输入层到隐藏层的Δ
    delta1 = delta1 / m
    delta2 = delta2 / m
    return serialize(delta1, delta2)
d1, d2 = deserialize(gradient(theta, X, y))

def regularized_gradient(theta, X, y, l=1):
    m = X.shape[0]
    delta1, delta2 = deserialize(gradient(theta, X, y))
    t1, t2 = deserialize(theta)
    t1[:, 0] = 0    #不把输入层到隐藏层的常数项引入正则化惩罚中
    reg_term_d1 = (l / m) * t1
    delta1 = delta1 + reg_term_d1#delta1就是Δ1
    t2[:, 0] = 0    #不把隐藏层到输出层的常数项引入正则化惩罚中
    reg_term_d2 = (l / m) * t2
    delta2 = delta2 + reg_term_d2
    return serialize(delta1, delta2)
def random_init(size): #初始化theta参数 接近0即可
    return np.random.uniform(-0.12, 0.12, size)
def nn_training(X, y): #梯度下降
    init_theta = random_init(10285)  # 25*401 + 10*26
    res = opt.minimize(fun=regularized_cost,
                       x0=init_theta,
                       args=(X, y, 1),
                       method='TNC',
                       jac=regularized_gradient,
                       options={'maxiter': 1000})
    return res
res = nn_training(X, y)
final_theta = res.x
print(res.fun)
_, y_answer = load_data("G:\pycharm_text\data_base\ex4data1.mat")
def show_accuracy(theta, X, y):
    _, _, _, _, h = feed_forward(theta, X)
    y_pred = np.argmax(h, axis=1) + 1
    y = y.reshape(5000,1)
    y_pred = y_pred.reshape(5000, 1)
    print(classification_report(np.matrix(y), np.matrix(y_pred)))
    correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y)]
    accuracy = (sum(map(int, correct)) / float(len(correct)))
    print('accuracy = {0}%'.format(accuracy * 100))

show_accuracy(final_theta,X,y_answer)


def plot_hidden_layer(theta):  #显示隐藏层
    final_theta1, _ = deserialize(theta)
    hidden_layer = final_theta1[:, 1:]  # ger rid of bias term theta
    fig, ax_array = plt.subplots(nrows=5, ncols=5, sharey=True, sharex=True, figsize=(5, 5))
    for r in range(5):
        for c in range(5):
            ax_array[r, c].matshow(hidden_layer[5 * r + c].reshape((20, 20)),
                                   cmap=matplotlib.cm.binary)
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
plot_hidden_layer(final_theta)
plt.show()














