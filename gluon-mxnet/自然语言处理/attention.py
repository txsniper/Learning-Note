import collections
import io
import mxnet as mx 
from mxnet import autograd, gluon, init, nd
from mxnet.contrib import text
from mxnet.gluon import data as gdata, loss as gloss, nn, rnn

PAD = '<pad>'
BOS = '<bos>'
EOS = '<eos>'

epochs = 40
epoch_period = 10
lr = 0.005
batch_size = 4
max_seq_len = 5
max_test_output_len = 20
encoder_num_layers = 1
decoder_num_layers = 2
encoder_drop_prob = 0.1
decoder_drop_prob = 0.1

encoder_num_hiddens = 256
decoder_num_hiddens = 256
alignment_size = 25
ctx = mx.cpu(0)

def read_data(max_seq_len):
    input_tokens = []
    output_tokens = []
    input_seqs = []
    output_seqs = []
    with io.open('./fr-en-small.txt') as f:
        lines = f.readlines()
        for line in lines:
            input_seq, output_seq = line.rstrip().split('\t')
            # 输入是法语单词
            cur_input_tokens = input_seq.split(' ')
            # 输出是英语单词
            cur_output_tokens = output_seq.split(' ')

            # 最长为5个单词
            if len(cur_input_tokens) < max_seq_len and len(cur_output_tokens) < max_seq_len:
                input_tokens.extend(cur_input_tokens)
                # 句末附上 EOS 符号。
                cur_input_tokens.append(EOS)
                # 添加 PAD 符号使每个序列等长（长度为 max_seq_len ）。
                while len(cur_input_tokens) < max_seq_len:
                    cur_input_tokens.append(PAD)
                # 输入句子列表
                input_seqs.append(cur_input_tokens)


                output_tokens.extend(cur_output_tokens)
                cur_output_tokens.append(EOS)
                while len(cur_output_tokens) < max_seq_len:
                    cur_output_tokens.append(PAD)
                # 输出句子列表
                output_seqs.append(cur_output_tokens)
        fr_vocab = text.vocab.Vocabulary(collections.Counter(input_tokens),
                                         reserved_tokens=[PAD, BOS, EOS])
        en_vocab = text.vocab.Vocabulary(collections.Counter(output_tokens),
                                         reserved_tokens=[PAD, BOS, EOS])
    return fr_vocab, en_vocab, input_seqs, output_seqs


input_vocab, output_vocab, input_seqs, output_seqs = read_data(max_seq_len)
# n * m : n为句子数量， m为句子长度
fr = nd.zeros((len(input_seqs), max_seq_len), ctx=ctx)
en = nd.zeros((len(output_seqs), max_seq_len), ctx=ctx)
for i in range(len(input_seqs)):
    fr[i] = nd.array(input_vocab.to_indices(input_seqs[i]), ctx=ctx)
    en[i] = nd.array(output_vocab.to_indices(output_seqs[i]), ctx=ctx)
dataset = gdata.ArrayDataset(fr, en)


class Encoder(nn.Block):
    def __init__(self, num_inputs, num_hiddens, num_layers, drop_prob, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        with self.name_scope():
            # 词向量
            self.embedding = nn.Embedding(num_inputs, num_hiddens)
            self.dropout = nn.Dropout(drop_prob)
            self.rnn = rnn.GRU(num_hiddens, num_layers, dropout=drop_prob,
                               input_size=num_hiddens)

    def forward(self, inputs, state):
        # print(inputs)
        # inputs: (batch_size, step_size)
        embedding = self.embedding(inputs)

        # embedding: (batch_size, step_size, num_hiddens)
        #print(embedding.shape)
        
        # swapaxes之后: embedding: (step_size, batch_size, num_hiddens)
        embedding = embedding.swapaxes(0, 1)
        #print(embedding.shape)
        embedding = self.dropout(embedding)

        # GRU输入数据格式: (sequence_length, batch_size, input_size)
        output, state = self.rnn(embedding, state)
        #print(output.shape)
        return output, state

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)


    
class Decoder(nn.Block):
    def __init__(self, num_hiddens, num_outputs, num_layers, max_seq_len,
                    drop_prob, alignment_size, encoder_num_hiddens, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.max_seq_len = max_seq_len
        self.encoder_num_hiddens = encoder_num_hiddens
        self.hidden_size = num_hiddens
        self.num_layers = num_layers
        with self.name_scope():
            # Decoder的输入是y，这里首先将y转化为词向量
            self.embedding = nn.Embedding(num_outputs, num_hiddens)
            self.dropout = nn.Dropout(drop_prob)

            # 注意力模块
            self.attention = nn.Sequential()
            with self.attention.name_scope():
                self.attention.add(
                    # 注意力第一层:
                    # 输入： num_hiddens(Decoder隐含状态) + encoder_num_hiddens(Encoder隐含状态)
                    # 输出： alignment_size(对齐)
                    nn.Dense(alignment_size,
                        in_units=num_hiddens + encoder_num_hiddens,
                        activation="tanh", flatten=False))
                    
                self.attention.add(
                    # 注意力第二层:
                    # 输入：alignment_size
                    # 输出：1
                    nn.Dense(1, in_units=alignment_size,
                                            flatten=False))

            self.rnn = rnn.GRU(num_hiddens, num_layers, dropout=drop_prob,
                               input_size=num_hiddens)
            self.out = nn.Dense(num_outputs, in_units=num_hiddens,
                                flatten=False)
            self.rnn_concat_input = nn.Dense(
                    num_hiddens, in_units=num_hiddens + encoder_num_hiddens,
                    flatten=False)

    # cur_input: 当前输入，batch_size个
    #    
    def forward(self, cur_input, state, encoder_outputs):
        print(state)
        # 当 RNN 为多层时，取最靠近输出层的单层隐藏状态。
        single_layer_state = [state[0][-1].expand_dims(0)]
        encoder_outputs = encoder_outputs.reshape((self.max_seq_len, -1,
                                                   self.encoder_num_hiddens))
        print(encoder_outputs.shape)
        hidden_broadcast = nd.broadcast_axis(single_layer_state[0], axis=0,
                                             size=self.max_seq_len)
        encoder_outputs_and_hiddens = nd.concat(encoder_outputs,
                                                hidden_broadcast, dim=2)
        energy = self.attention(encoder_outputs_and_hiddens)
        batch_attention = nd.softmax(energy, axis=0).transpose((1, 2, 0))
        batch_encoder_outputs = encoder_outputs.swapaxes(0, 1)
        decoder_context = nd.batch_dot(batch_attention, batch_encoder_outputs)
        input_and_context = nd.concat(
            nd.expand_dims(self.embedding(cur_input), axis=1),
            decoder_context, dim=2)
        concat_input = self.rnn_concat_input(input_and_context).reshape(
            (1, -1, 0))
        concat_input = self.dropout(concat_input)
        state = [nd.broadcast_axis(single_layer_state[0], axis=0,
                                   size=self.num_layers)]
        output, state = self.rnn(concat_input, state)
        output = self.dropout(output)
        print(output.shape)
        output = self.out(output).reshape((-3, -1))
        return output, state

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)

    
class DecoderInitState(nn.Block):
    """解码器隐藏状态的初始化。"""
    def __init__(self, encoder_num_hiddens, decoder_num_hiddens, **kwargs):
        super(DecoderInitState, self).__init__(**kwargs)
        with self.name_scope():
            self.dense = nn.Dense(decoder_num_hiddens,
                                  in_units=encoder_num_hiddens,
                                  activation="tanh", flatten=False)

    def forward(self, encoder_state):
        return [self.dense(encoder_state)]


def translate(encoder, decoder, decoder_init_state, fr_ens, ctx, max_seq_len):
    for fr_en in fr_ens:
        print('[input] ', fr_en[0])
        input_tokens = fr_en[0].split(' ') + [EOS]
        # 添加 PAD 符号使每个序列等长（长度为 max_seq_len）。
        while len(input_tokens) < max_seq_len:
            input_tokens.append(PAD)
        inputs = nd.array(input_vocab.to_indices(input_tokens), ctx=ctx)
        encoder_state = encoder.begin_state(func=nd.zeros, batch_size=1,
                                            ctx=ctx)
        encoder_outputs, encoder_state = encoder(inputs.expand_dims(0),
                                                 encoder_state)
        encoder_outputs = encoder_outputs.flatten()
        # 解码器的第一个输入为 BOS 字符。
        decoder_input = nd.array([output_vocab.token_to_idx[BOS]], ctx=ctx)
        decoder_state = decoder_init_state(encoder_state[0])
        output_tokens = []

        for _ in range(max_test_output_len):
            decoder_output, decoder_state = decoder(
                decoder_input, decoder_state, encoder_outputs)
            pred_i = int(decoder_output.argmax(axis=1).asnumpy()[0])
            # 当任一时刻的输出为 EOS 字符时，输出序列即完成。
            if pred_i == output_vocab.token_to_idx[EOS]:
                break
            else:
                output_tokens.append(output_vocab.idx_to_token[pred_i])
            decoder_input = nd.array([pred_i], ctx=ctx)
        print('[output]', ' '.join(output_tokens))
        print('[expect]', fr_en[1], '\n')


loss = gloss.SoftmaxCrossEntropyLoss()
eos_id = output_vocab.token_to_idx[EOS]

def train(encoder, decoder, decoder_init_state, max_seq_len, ctx,
          eval_fr_ens):
    encoder.initialize(init.Xavier(), ctx=ctx)
    decoder.initialize(init.Xavier(), ctx=ctx)
    decoder_init_state.initialize(init.Xavier(), ctx=ctx)
    encoder_optimizer = gluon.Trainer(encoder.collect_params(), 'adam',
                                      {'learning_rate': lr})
    decoder_optimizer = gluon.Trainer(decoder.collect_params(), 'adam',
                                      {'learning_rate': lr})
    decoder_init_state_optimizer = gluon.Trainer(
        decoder_init_state.collect_params(), 'adam', {'learning_rate': lr})

    data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)
    l_sum = 0
    for epoch in range(1, epochs + 1):
        for x, y in data_iter:
            cur_batch_size = x.shape[0]
            with autograd.record():
                l = nd.array([0], ctx=ctx)
                valid_length = nd.array([0], ctx=ctx)

                # encoder状态尺寸： （num_layers， batch_size, hidden_dim)
                encoder_state = encoder.begin_state(
                    func=nd.zeros, batch_size=cur_batch_size, ctx=ctx)
                print(encoder_state)
                
                encoder_outputs, encoder_state = encoder(x, encoder_state)

                # For an input array with shape (d1, d2, ..., dk), flatten operation reshapes the input array into # an output array of shape (d1, d2*...*dk)
                # flatten将矩阵变换为二维的，将每个
                print(encoder_outputs.shape)
                encoder_outputs = encoder_outputs.flatten()
                print(encoder_outputs.shape)
                
                # 解码器的第一个输入为 BOS 字符。
                decoder_input = nd.array(
                    [output_vocab.token_to_idx[BOS]] * cur_batch_size,
                    ctx=ctx)
                print(decoder_input)
            
                mask = nd.ones(shape=(cur_batch_size,), ctx=ctx)

                # encoder_state是一个list,其中的每个元素是一个()
                print(encoder_state[0].shape)
                print(len(encoder_state))
                decoder_state = decoder_init_state(encoder_state[0])
                exit(0)
                print(decoder_state)
                for i in range(max_seq_len):
                    decoder_output, decoder_state = decoder(
                        decoder_input, decoder_state, encoder_outputs)
                    # 解码器使用当前时刻的预测结果作为下一时刻的输入。
                    decoder_input = decoder_output.argmax(axis=1)
                    valid_length = valid_length + mask.sum()
                    l = l + (mask * loss(decoder_output, y[:, i])).sum()
                    mask = mask * (y[:, i] != eos_id)
                l = l / valid_length
            l.backward()
            encoder_optimizer.step(1)
            decoder_optimizer.step(1)
            decoder_init_state_optimizer.step(1)
            l_sum += l.asscalar() / max_seq_len

        if epoch % epoch_period == 0 or epoch == 1:
            if epoch == 1:
                print('epoch %d, loss %f, ' % (epoch, l_sum / len(data_iter)))
            else:
                print('epoch %d, loss %f, '
                      % (epoch, l_sum / epoch_period / len(data_iter)))
            if epoch != 1:
                l_sum = 0
            translate(encoder, decoder, decoder_init_state, eval_fr_ens, ctx,
                      max_seq_len)


encoder = Encoder(len(input_vocab), encoder_num_hiddens, encoder_num_layers, 
                    encoder_drop_prob)
decoder = Decoder(decoder_num_hiddens, len(output_vocab),
                  decoder_num_layers, max_seq_len, decoder_drop_prob,
                  alignment_size, encoder_num_hiddens)
decoder_init_state = DecoderInitState(encoder_num_hiddens,
                                      decoder_num_hiddens)


eval_fr_ens =[['elle est japonaise .', 'she is japanese .'],
              ['ils regardent .', 'they are watching .']]
train(encoder, decoder, decoder_init_state, max_seq_len, ctx, eval_fr_ens)
