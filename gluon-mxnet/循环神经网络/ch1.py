import zipfile
with zipfile.ZipFile('./jaychou_lyrics.txt.zip', 'r') as zin:
    zin.extractall('./data/')

with open('./data/jaychou_lyrics.txt') as f:
    corpus_chars = f.read()
#print(corpus_chars[0:49])

corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
corpus_chars = corpus_chars[0:20000]
idx_to_char = list(set(corpus_chars))
char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
vocab_size = len(char_to_idx)
print('vocab size:', vocab_size)

corpus_indices = [char_to_idx[char] for char in corpus_chars]
sample = corpus_indices[:40]
print('chars:\n', ''.join([idx_to_char[idx] for idx in sample]))
print('\nindices:\n', sample)

import random
from mxnet import nd

# 随机批量采样
def data_iter_random(corpus_indices, batch_size, num_steps, ctx=None):
    # 总样本数量
    num_examples = (len(corpus_indices) - 1)
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

my_seq = list(range(30))
for data, label in data_iter_random(my_seq, batch_size=2, num_steps=3):
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

