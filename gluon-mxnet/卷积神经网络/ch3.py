from mxnet import nd, autograd
from mxnet.gluon import nn

def corr2d(X, K):
    n, m = K.shape
    Y = nd.zeros((X.shape[0] - n + 1, X.shape[1] - m + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i+n, j:j+m]*K).sum()
    return Y

def corr2d_multi_in(X, K):
    return nd.add_n(*[corr2d(x, k) for x, k in zip(X, K)])

# X:双通道，每个通道
X = nd.array([
            [[0,1,2],[3,4,5],[6,7,8]],
            [[1,2,3],[4,5,6],[7,8,9]]
            ])
K = nd.array([
            [[0,1], [2,3]],
            [[1,2], [3,4]]
            ])
corr2d_multi_in(X, K)