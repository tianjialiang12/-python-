import numpy as np
import scipy.io as sio
from sklearn.metrics import classification_report

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


def load_weight(path):
    data = sio.loadmat(path)
    return data['Theta1'], data['Theta2']
theta1, theta2 = load_weight("G:\pycharm_text\data_base\ex3weights.mat")
print(theta1.shape, theta2.shape)#theta1 就是输入层到隐藏层的theta值 隐藏层有25个神经元（加一个偏置项）
#这里theta11 theta2 都是已经训练好的 #theta2 是隐藏层到输出层的theta值  输出层有10个神经元
X, y = load_data("G:\pycharm_text\data_base\ex3data1.mat",transpose=False) #因为以上theta值是在原始数据中训练得到的 所以在此不能转置
X = np.insert(X, 0, values=np.ones(X.shape[0]), axis=1)  # intercept
#print(X.shape, y.shape)

########################          前向传播      ######################
a1 = X                #输入层的a就是x
z2 = a1 @ theta1.T    #隐藏层的z  g（z） 和逻辑回归一样 z= X * theta
z2 = np.insert(z2, 0, values=np.ones(z2.shape[0]), axis=1)
a2 = sigmoid(z2)
z3 = a2 @ theta2.T
a3 = sigmoid(z3)
y_pred = np.argmax(a3, axis=1) + 1     #返回每一行最大数的索引 即预测值接近1的那个数
print(classification_report(y, y_pred))