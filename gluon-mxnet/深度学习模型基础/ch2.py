from mxnet import ndarray as nd 
from mxnet import autograd
from mxnet import gluon

num_inputs = 2
num_examples = 1000

true_w = [2, -3.4]
true_b = 4.2

# step1: 生成数据集
X = nd.random_normal(shape=(num_examples, num_inputs))
y = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_b
y += 0.01 * nd.random_normal(shape=y.shape)

# step2: 利用gluon来分批次读取数据
batch_size = 10
dataset = gluon.data.ArrayDataset(X, y)
data_iter = gluon.data.DataLoader(dataset, batch_size, shuffle=True)

for data, label in data_iter:
    print(data, label)
    break

# step3: 构建net
net = gluon.nn.Sequential()
net.add(gluon.nn.Dense(1))
net.initialize()
square_loss = gluon.loss.L2Loss()
trainer = gluon.Trainer(
    net.collect_params(), 'sgd', {'learning_rate': 0.1}
)

epochs = 5
batch_size = 10
for e in range(epochs):
    total_loss = 0
    for data, label in data_iter:
        with autograd.record():
            output = net(data)
            loss = square_loss(output, label)
        loss.backward()

        # 更新一次梯度，gluon会把计算出来的梯度除以batch_size
        trainer.step(batch_size)
        total_loss += nd.sum(loss).asscalar()
    print("Epoch %d, avg loss: %f" % (e, total_loss/num_examples))

dense = net[0]
print("true_w : " + str(true_w))
print("net weight: " + str(dense.weight.data()))
print("true_b : " + str(true_b))
print("net bias: " + str(dense.bias.data()))
