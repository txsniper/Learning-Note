#模型参数的访问，初始化和共享

from mxnet import init, nd 
from mxnet.gluon import nn 

net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()

x = nd.random.uniform(shape=(2, 20))
y = net(x)

print(net[0].bias.data())


class MyInit(init.Initializer):
    def __init_weight(self, name, data):
        print('Init', name, data.shape)
        data[:] = nd.random.uniform(low=-10, high=10, shape=data.shape)
        data *= data.abs() >= 5

#net.initialize(MyInit())
#net.initialize(MyInit(), force_reinit=True)
#print(net[0].weight.data()[0])


from mxnet import nd
from mxnet.gluon import nn 
net = nn.Sequential()
shared = nn.Dense(8, activation='relu')
net.add(nn.Dense(8, activation='relu'),
        shared,
        nn.Dense(8, activation='relu', params=shared.params),
        nn.Dense(10))
net.initialize()

x = nd.random.uniform(shape=(2, 20))
net(x)

print(net[1].weight.data()[0] == net[2].weight.data()[0])