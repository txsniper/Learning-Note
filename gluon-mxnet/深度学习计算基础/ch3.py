from mxnet import init, nd 
from mxnet.gluon import nn 

class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        print('Init', name, data.shape)

net = nn.Sequential()

# output: 256维
net.add(nn.Dense(256, activation='relu'))

# output: 10维
net.add(nn.Dense(10))

net.initialize(init=MyInit())
x = nd.random.uniform(shape=(2, 20))
print(x)
y = net(x)
