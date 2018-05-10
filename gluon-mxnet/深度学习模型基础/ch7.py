# 欠拟合与过拟合

from mxnet import ndarray as nd 
from mxnet import autograd
from mxnet import gluon

num_train = 100
num_test  = 100
true_w = [1.2, -3.4, 5.6]
true_b = 5.0

# x : 200 * 1
x = nd.random.normal(shape=(num_train + num_test, 1))

# X : 200 * 3
X = nd.concat(x, nd.power(x, 2), nd.power(x, 3))

# y = a * x + b * x^2 + c * x^3 + d 
y = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_w[2] * X[:, 2] + true_b


