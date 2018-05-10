# dropout从零开始

from mxnet import nd

# 丢弃的方式就是将其中的元素置为0
def dropout(X, drop_prob):
    keep_prob = 1 - drop_prob
    assert 0 <= keep_prob <= 1

    # 全部置为0
    if keep_prob == 0:
        return X.zeros_like()

    # 这里从【0， 1】均匀分布中采样
    # ctx : 用来设置上下文（比如使用GPU),这里与X的context一致
    temp = nd.random.uniform(
        0, 1.0, X.shape, ctx=X.context
    )
    #print(temp)

    # 生成 mask
    mask = temp < keep_prob
    scale = 1 / keep_prob
    return mask * X * scale

'''
A = nd.arange(20).reshape((5,4))
print(A)
B = dropout(A, 0.5)
print(B)
'''

import sys
sys.path.append('..')
import utils
batch_size = 256
train_data, test_data = utils.load_data_fashion_mnist(batch_size)

num_input = 28 * 28
num_outputs = 10

num_hidden1 = 256
num_hidden2 = 256
weigth_scale = 0.01

W1 = nd.random_normal(shape=(num_input, num_hidden1), scale=weigth_scale)
b1 = nd.zeros(num_hidden1)

W2 = nd.random_normal(shape=(num_hidden1, num_hidden2), scale=weigth_scale)
b2 = nd.zeros(num_hidden2)

W3 = nd.random_normal(shape=(num_hidden2, num_outputs), scale=weigth_scale)
b3 = nd.zeros(num_outputs)

params = [W1, b1, W2, b2, W3, b3]
for param in params:
    param.attach_grad()

drop_prob1 = 0.5
drop_prob2 = 0.2

def net(X):
    X = X.reshape((-1, num_input))
    h1 = nd.relu(nd.dot(X, W1) + b1)
    h1 = dropout(h1, drop_prob1)
    h2 = nd.relu(nd.dot(h1, W2) + b2)
    h2 = dropout(h2, drop_prob2)
    return nd.dot(h2, W3) + b3

from mxnet import autograd
from mxnet import gluon
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
learning_rate = 0.5

for epoch in range(30):
    train_loss = 0.0
    train_acc  = 0.0
    for data, label in train_data:
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        utils.SGD(params, learning_rate/batch_size)
        train_loss += nd.mean(loss).asscalar()
        train_acc  += utils.accuracy(output, label)
    test_acc = utils.evaluate_accuracy(test_data, net)
    print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
        epoch, train_loss/len(train_data), train_acc/len(train_data), test_acc
    ))
