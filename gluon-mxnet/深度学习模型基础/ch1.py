from mxnet import ndarray as nd
from mxnet import autograd
import matplotlib.pyplot as plt

num_inputs = 2
num_examples = 1000

true_w = [2, -3.4]
true_b = 4.2

X = nd.random_normal(shape=(num_examples, num_inputs))
y = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_b
y += 0.01 * nd.random_normal(shape=y.shape)

#plt.scatter(X[:, 1].asnumpy(), y.asnumpy())
#plt.show()


import random

batch_size = 10

def data_iter():
    idx = list(range(num_examples))
    random.shuffle(idx)
    for i in range(0, num_examples, batch_size):
        #print("i = " + str(i))
        j = nd.array(idx[i : min(i+batch_size, num_examples)])
        #print(j)
        yield nd.take(X, j), nd.take(y, j)

def test_data_iter():
    for data, label  in data_iter():
        print(data, label)
        break

w = nd.random_normal(shape=(num_inputs, 1))
b = nd.zeros((1, ))
params = [w, b]

# step1 : 调用 attach_grad申请内存存储param的梯度
for param in params:
    param.attach_grad()


def net(X):
    return nd.dot(X, w) + b

def square_loss(yhat, y):
    return (yhat - y.reshape(yhat.shape)) ** 2

def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad

def real_fn(X):
    return true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_b

def plot(losses, X, sample_size=100):
    xs = list(range(len(losses)))
    f, (fg1, fg2) = plt.subplots(1, 2)
    fg1.set_title('Loss during training')
    fg1.plot(xs, losses, '-r')
    fg2.set_title('Estimated vs real function')
    fg2.plot(X[:sample_size, 1].asnumpy(),
            net(X[:sample_size, : ]).asnumpy(), 'or', label='Estimated')
    fg2.plot(X[:sample_size, 1].asnumpy(),
            real_fn(X[:sample_size, : ]).asnumpy(), '*g', label='Real')
    #fg2.legend()
    plt.show()


epochs = 5
learning_rate = 0.001
niter = 0
temp_iter = 0
losses = []
moving_loss = 0
smoothing_constant = 0.01

for e in range(epochs):
    total_loss = 0
    print('temp_iter = ' + str(temp_iter))
    temp_iter += 1
    for data, label in data_iter():
        # step2: 利用 record 记录梯度相关的计算
        with autograd.record():
            output = net(data)
            loss = square_loss(output, label)
        # step3: 求解梯度
        loss.backward()

        # step4: 利用梯度更新参数
        SGD(params, learning_rate)

        # asscalar() 将一个值的数组转换为标量
        total_loss += nd.sum(loss).asscalar()

        niter += 1
        # print(str(niter) + str(loss))
        curr_loss = nd.mean(loss).asscalar()
        moving_loss = (1 - smoothing_constant) * moving_loss + (smoothing_constant) * curr_loss
        est_loss = moving_loss / (1-(1-smoothing_constant)**niter)

        if (niter + 1) % 100 == 0:
            losses.append(est_loss)
            print("Epoch %s, batch %s. Moving avg of loss: %s. Average loss:%f" % (e, niter, est_loss, total_loss/num_examples))
            plot(losses, X)

