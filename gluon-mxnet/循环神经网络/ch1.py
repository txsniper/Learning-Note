import zipfile
with zipfile.ZipFile('./jaychou_lyrics.txt.zip', 'r') as zin:
    zin.extractall('./data/')

with open('./data/jaychou_lyrics.txt') as f:
    corpus_chars = f.read()
#print(corpus_chars[0:49])

corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
corpus_chars = corpus_chars[0:60]
idx_to_char = list(set(corpus_chars))
char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
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
    print(len(corpus_indices))
    print(num_steps)
    print(num_examples)
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
