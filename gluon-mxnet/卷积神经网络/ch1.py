# 卷积层
import matplotlib.pyplot as plt
from mxnet import nd, autograd
from mxnet.gluon import nn

def corr2d(X, K):
    n, m = K.shape
    Y = nd.zeros((X.shape[0]-n+1, X.shape[1]-m+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i+n, j:j+m]*K).sum()
    return Y

X = nd.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
K = nd.array([[0, 1], [2, 3]])
#print(corr2d(X, K))

X = nd.ones((6, 8))
X[:, 2:6] = 0
print(X)
plt.imshow(X.asnumpy(), cmap='gray')
K = nd.array([[1, -1]])
#print(K)
Y = corr2d(X, K)
print(Y)

class Conv2D(nn.Block):
    def __init__(self, kernel_size, **kwargs):
        super(Conv2D, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=kernel_size)
        self.bias = self.params.get('bias', shape=(1, ))

    def forward(self, x):
        return corr2d(x, self.weight.data()) + self.bias.data()

conv2d = nn.Conv2D(1, kernel_size=(1, 2))
conv2d.initialize()

X = X.reshape((1, 1, 6, 8))
#print(X)
Y = Y.reshape((1, 1, 6, 7))

for i in range(30):
    with autograd.record():
        pY = conv2d(X)
        #print(pY)
        loss = (pY - Y) ** 2
        print('batch %d, loss %.3f' % (i, loss.sum().asscalar()))
    loss.backward()
    conv2d.weight.data()[:] -= 3e-2 * conv2d.weight.grad()
print(conv2d.weight.data())