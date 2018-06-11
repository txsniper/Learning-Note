import zipfile
with zipfile.ZipFile('./jaychou_lyrics.txt.zip', 'r') as zin:
    zin.extractall('./data/')

with open('./data/jaychou_lyrics.txt') as f:
    corpus_chars = f.read()
#print(corpus_chars[0:49])

corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
corpus_chars = corpus_chars[0:20000]

# 构建一个字符列表
idx_to_char = list(set(corpus_chars))

# 字符字典：char --> index
char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])

# 字符总数量
vocab_size = len(char_to_idx)
print('vocab size:', vocab_size)

corpus_indices = [char_to_idx[char] for char in corpus_chars]
sample = corpus_indices[:5]
print('chars:\n', ''.join([idx_to_char[idx] for idx in sample]))
print('\nindices:\n', sample)

import random
from mxnet import nd

# 随机批量采样
def data_iter_random(corpus_indices, batch_size, num_steps, ctx=None):
    # 总样本数量
    num_examples = (len(corpus_indices) - 1) // num_steps
    #print(len(corpus_indices))
    #print(num_steps)
    #print(num_examples)
    epoch_size = num_examples // batch_size

    # 下标随机
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)

    def _data(pos):
        return corpus_indices[pos: pos + num_steps]
    
    for i in range(epoch_size):
        i = i * batch_size
        batch_indices = example_indices[i : i+batch_size]
        data = nd.array(
            [_data(j * num_steps) for j in batch_indices], ctx=ctx
        )
        label = nd.array(
            [_data(j * num_steps + 1) for j in batch_indices], ctx=ctx
        )
        yield data, label

my_seq = list(range(5))
for data, label in data_iter_random(my_seq, batch_size=2, num_steps=2):
    print('data: ', data, '\nlabel:', label, '\n')

# 相邻批量采样
def data_iter_consecutive(corpus_indices, batch_size, num_steps, ctx=None):
    corpus_indices = nd.array(corpus_indices, ctx=ctx)
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size
    indices = corpus_indices[0: batch_size * batch_len].reshape((
        batch_size, batch_len
    ))
    epoch_size = (batch_len - 1) // num_steps
    for i in range(epoch_size):
        i = i * num_steps
        data = indices[:, i:i+num_steps]
        label = indices[:, i+1 : i+num_steps+1]
        yield data, label

my_seq = list(range(30))
for data, label in data_iter_consecutive(my_seq, batch_size=2, num_steps=3):
    print('data: ', data, '\nlabel:', label, '\n')


def get_inputs(data):
    return [nd.one_hot(X, vocab_size) for X in data.T]

print(data)
inputs = get_inputs(data)
print('input length:', len(inputs))
print('input shape: ', inputs)
print('input shape: ', inputs[0])


import mxnet as mx
import sys 
sys.path.append('..')
import gluonbook as gb
ctx = gb.try_gpu()
print('Will use', ctx)
input_dim = vocab_size
hidden_dim = 256
output_dim = vocab_size
std = 0.01

def get_params():
    W_xh = nd.random_normal(scale=std, shape=(input_dim, hidden_dim), ctx=ctx)
    W_hh = nd.random_normal(scale=std, shape=(hidden_dim, hidden_dim), ctx=ctx)
    b_h = nd.zeros(hidden_dim, ctx=ctx)

    W_hy = nd.random_normal(scale=std, shape=(hidden_dim, output_dim), ctx=ctx)
    b_y  = nd.zeros(output_dim, ctx=ctx)

    params = [W_xh, W_hh, b_h, W_hy, b_y]
    for param in params:
        param.attach_grad()
    return params

def rnn(inputs, state, *params):
    H = state
    W_xh, W_hh, b_h, W_hy, b_y = params
    outputs = []
    for X in inputs:
        H = nd.tanh(nd.dot(X, W_xh) + nd.dot(H, W_hh) + b_h)
        Y = nd.dot(H, W_hy) + b_y
        outputs.append(Y)
    return (outputs, H)

state = nd.zeros(shape=(data.shape[0], hidden_dim), ctx=ctx)
params = get_params()
outputs, state_new = rnn(get_inputs(data.as_in_context(ctx)), state, *params)
print('output length: ', len(outputs))
print('output[0] shape: ', outputs[0].shape)
print('state shape: ', state_new.shape)


# 预测以 prefix 开始接下来 num_chars个字符
def predict_rnn(rnn, prefix, num_chars, params, hidden_dim, ctx, idx_to_char,
                char_to_idx, get_inputs, is_lstm=False):
    prefix = prefix.lower()
    state_h = nd.zeros(shape=(1, hidden_dim), ctx=ctx)
    if is_lstm:
        state_c = nd.zeros(shape=(1, hidden_dim), ctx=ctx)
    output = [char_to_idx[prefix[0]]]
    for i in range(num_chars + len(prefix)):
        X = nd.array([output[-1]], ctx=ctx)
        if is_lstm:
            Y, state_h, state_c = rnn(get_inputs(X), state_h, state_c, *params)
        else:
            Y, state_h = rnn(get_inputs(X), state_h, *params)
        if i < len(prefix) - 1:
            next_input = char_to_idx[prefix[i+1]]
        else:
            next_input = int(Y[0].argmax(axis=1).asscalar())
        output.append(next_input)
    return ''.join([idx_to_char[i] for i in output])

def grad_clipping(params, theta, ctx):
    if theta is not None:
        norm = nd.array([0.0], ctx)
        for p in params:
            norm += nd.sum(p.grad ** 2)
        norm = nd.sqrt(norm).asscalar()
        if norm > theta:
            for p in params:
                p.grad[:] *= theta / norm

from mxnet import autograd
from mxnet import gluon
from math import exp

def train_and_predict_rnn(rnn, is_random_iter, epochs, num_steps, hidden_dim,
                        learning_rate, clipping_theta, batch_size, pred_period,
                        pred_len, seqs, get_params, get_inputs, ctx, corpus_indices,
                        idx_to_char, char_to_idx, is_lstm=False):
    if is_random_iter:
        data_iter = data_iter_random
    else:
        data_iter = data_iter_consecutive
    params = get_params()
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    for e in range(1, epochs + 1):
        # 如果使用相邻批量采样，同一个epoch中，隐含变量只需要在开始初始化
        if not is_random_iter:
            # 每一个字符的中间状态维度为hidden_dim，每次有batch_size个字符
            state_h = nd.zeros(shape=(batch_size, hidden_dim), ctx=ctx)
            if is_lstm:
                state_c = nd.zeros(shape=(batch_size, hidden_dim), ctx=ctx)
        train_loss, num_examples = 0, 0
        for data, label in data_iter(corpus_indices, batch_size, num_steps, ctx):
            # 如果使用随机批量采样，处理每个之前都需要初始化隐含变量
            if is_random_iter:
                state_h = nd.zeros(shape=(batch_size, hidden_dim), ctx=ctx)
                if is_lstm:
                    state_c = nd.zeros(shape=(batch_size, hidden_dim), ctx=ctx)

            with autograd.record():
                # outputs 尺寸 ：batch_size, vocab_size
                if is_lstm:
                    outputs, state_h, state_c = rnn(get_inputs(data), state_h, state_c, *params)
                else:
                    outputs, state_h = rnn(get_inputs(data), state_h, *params)
                # 设 t_ib_j为i时间批量中的j元素:
                # label尺寸： batch_size * num_steps
                # label = [t_0b_0, t_0b_1, ... , t_1b_0, t_1b_1, ... ,]
                label = label.T.reshape((-1,))
                # 拼接 outputs,尺寸 : batch_size * num_steps, vocab_size
                outputs = nd.concat(*outputs, dim=0)
                loss = softmax_cross_entropy(outputs, label)
            loss.backward()
            grad_clipping(params, clipping_theta, ctx)
            gb.SGD(params, learning_rate)
            train_loss += nd.sum(loss).asscalar()
            num_examples += loss.size
        if e % pred_period == 0:
            print("Epoch %d. Perplexity %f" % (e, exp(train_loss/num_examples)))
            for seq in seqs:
                print(' - ', predict_rnn(rnn, seq, pred_len, params,
                       hidden_dim, ctx, idx_to_char, char_to_idx, get_inputs, is_lstm))
            print()


epochs = 200
num_steps = 35
learning_rate = 0.1
batch_size = 32
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
seq1 = '分开'
seq2 = '不分开'
seq3 = '战争中部队'
seqs = [seq1, seq2, seq3]
train_and_predict_rnn(rnn=rnn, is_random_iter=True, epochs=epochs, num_steps=num_steps,
                    hidden_dim=hidden_dim, learning_rate=learning_rate,
                    clipping_theta=5, batch_size=batch_size, pred_period=20,
                    pred_len=100, seqs=seqs, get_params=get_params,
                    get_inputs=get_inputs, ctx=ctx,
                    corpus_indices=corpus_indices, idx_to_char=idx_to_char,
                    char_to_idx=char_to_idx)