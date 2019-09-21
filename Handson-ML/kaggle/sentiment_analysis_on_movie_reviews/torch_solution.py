import numpy as np 
import pandas as pd 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import functional as F 

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem import SnowballStemmer

import random
torch.set_num_threads(32)

class Util(object):
    @staticmethod
    def var_batch(batch_size, sents, sent_lens, sent_mask, max_len):
        #print("batch_size: " + str(len(sents)))
        '''
        for sent in sents:
            print("sent len : " + str(len(sent)))
        '''
        #print(sents[0:5])
        #print("NOTE")
        #print(sent_lens[0:5])
        sents_ = Variable(torch.LongTensor(sents).view(batch_size, max_len))
        sent_lens_ = Variable(torch.LongTensor(sent_lens))
        if sent_mask is not None:
            sent_mask_ = Variable(torch.LongTensor(sent_mask).view(batch_size, max_len))
        else:
            sent_mask_ = None
        return sents_, sent_lens_, sent_mask_
    
    @staticmethod
    def get_padding(sents, max_len):
        # np.zeros((0, max_len)) : 创建一个0行, max_len列的矩阵
        #seq_len = np.zeros((0,))
        #padded = np.zeros((0, max_len))
        padded = []
        seq_len = []
        for sent in sents:
            num_words = len(sent)
            num_pad = max_len - num_words
            if max_len == 60 and num_words > 60:
                sent = sent[:45] + sent[num_words - 15:]
                sent = np.asarray(sent, dtype=np.int64).reshape(1, -1)
            else:
                sent = np.asarray(sent[: max_len], dtype=np.int64).reshape(1, -1)
            if num_pad > 0:
                zero_paddings = np.zeros((1, num_pad), dtype=np.int64)
                sent = np.concatenate((sent, zero_paddings), axis=1)
            else:
                num_words = max_len
            
            # 保存每个padding之后的句子
            #padded = np.concatenate((padded, sent), axis=0) # axis=0, 按行连接
            padded.append(sent)
            #seq_len = np.concatenate((seq_len, [num_words]))
            #seq_len.append([num_words])
            seq_len.append(num_words)
            
        padded = np.asarray(padded).reshape(-1, max_len)
        print(padded.shape)
        seq_len = np.asarray(seq_len)
        return padded.astype(np.int64), seq_len.astype(np.int64)
    
    @staticmethod
    def get_mask_matrix(seq_lens, max_len):
        mask_matrix = np.ones((0, max_len))
        for seq_len in seq_lens:
            num_mask = max_len - seq_len
            mask = np.ones((1, seq_len), dtype=np.int64)
            if num_mask > 0:
                zero_paddings = np.zeros((1, num_mask), dtype=np.int64)
                mask = np.concatenate((mask, zero_paddings), axis=1)
            mask_matrix = np.concatenate((mask_matrix, mask), axis=0)
        return mask_matrix
    
    @staticmethod
    def cal_prf(pred, right, gold, formation=True, metrics_type=""):
        num_class = len(pred)
        precision = [0.0] * num_class
        recall = [0.0] * num_class
        f1_score = [0.0] * num_class
        
        for i in range(num_class):
            precision[i] = 0 if pred[i] == 0 else 1.0 * right[i] / pred[i]
            recall[i] = 0 if gold[i] == 0 else 1.0 * right[i] / gold[i]
            f1_score[i] = 0 if precision[i] == 0 or recall[i] == 0 else 2.0 * (precision[i] * recall[i]) / (precision[i] + recall[i])
            
            precision[i] = precision[i].__format__(".6f")
            recall[i] = recall[i].__format__(".6f")
            f1_score[i] = f1_score[i].__format__(".6f")
        
        '''
        if metrics_type == "macro":
            precision = sum(precision) / len(precision)
            recall = sum(recall) / len(recall)
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        elif metrics_type == "micro":
            precision = 1.0 * sum(right) / sum(pred) if sum(pred) > 0 else 0
            recall = 1.0 * sum(right) / sum(gold) if sum(recall) > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        '''   
        return precision, recall, f1_score

class LSTM_Model(nn.Module):
    def __init__(self, embeddings, input_dim, vocab_size, hidden_dim, num_layers, output_dim, max_len=100, dropout=0.5):
        super(LSTM_Model, self).__init__()
        if embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(
                embeddings,
                freeze=False
            )
        else:
            self.embedding = nn.Embedding(
                vocab_size,
                input_dim,
                padding_idx=0
            )
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            1,
            dropout=dropout, 
            bidirectional=False, 
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.output_dim = output_dim
        
    def forward(self, x, length, mask=None):
        batch_size = len(x)
        x_emb = self.embedding(x)
        lstm_output, _ = self.lstm(x_emb)
        output = self.fc(lstm_output[:,-1,:])
        return output
        
    '''
    def forward(self, x, length, mask=None):
        batch_size = len(x)
        # step1: sort x
        length_sorted, idx_sorted = torch.sort(length, descending=True)
        _, idx_unsort = torch.sort(idx_sorted)
        x_sorted = x[idx_sorted]
        
        #print(length.shape)
        #print(length)
        
        
        # step2: emb and pack and rnn
        x_emb = self.embedding(x_sorted)
        x_packed = nn.utils.rnn.pack_padded_sequence(x_emb, length_sorted, batch_first=True)
        lstm_output, (lstm_hidden, lstm_cell_state) = self.lstm(x_packed)
        #exit(255)
        
        # step3: unpack
        # padded_output: (batch_size, time_steps, hidden_size * bi_num)
        # h_n|c_n: (num_layer*bi_num, batch_size, hidden_size)
        lstm_output_pad, _ = nn.utils.rnn.pad_packed_sequence(lstm_output, batch_first=True)
        
        lstm_output_pad = lstm_output_pad[idx_unsort]
    
        # 取最后一个有效输出作为最终输出（0为无效输出)
        last_output = lstm_output_pad[torch.LongTensor(range(batch_size)), length - 1]
        output = self.fc(last_output)
        #print(output)
        #out_prob = F.softmax(output.view(batch_size, self.output_dim), dim=1)
        return output
    '''
        
class LSTM(nn.Module):
    def __init__(self, embeddings, input_dim, vocab_size, hidden_dim, num_layers, output_dim, max_len=40, dropout=0.5):
        super(LSTM, self).__init__()
        self.emb = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=input_dim,
            padding_idx=0
        )
        if embeddings is not None:
            self.emb.weight = torch.nn.Parameter(embeddings)
            self.emb.weight.requires_grad = True
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.sen_len = max_len
        self.sen_rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=False
        )
        self.output = nn.Linear(self.hidden_dim, output_dim)
    
    def _fetch(self, rnn_outs, seq_lengths, batch_size, max_len):
        rnn_outs = rnn_outs.view(batch_size, max_len, 1, -1)
        fw_out = torch.index_select(rnn_outs, 2, Variable(torch.LongTensor([0])))
        fw_out = fw_out.view(batch_size * max_len, -1)
        batch_range = Variable(torch.LongTensor(range(batch_size))) * max_len
        fw_index = batch_range + seq_lengths.view(batch_size) - 1
        fw_out = torch.index_select(fw_out, 0, fw_index)
        return fw_out
    
    def forward(self, sen_batch, sen_lengths, sen_mask_matrix):
        sen_batch = self.emb(sen_batch)
        batch_size = len(sen_batch)
        
        sen_outs, _ = self.sen_rnn(sen_batch.view(batch_size, -1, self.input_dim))
        
        # Batch_first only change viewpoint, may not be contiguous
        sen_rnn = sen_outs.contiguous().view(batch_size, -1, self.hidden_dim)  # (batch, sen_len, 2*hid)

        ''' Fetch the truly last hidden layer of both sides
        '''
        sentence_batch = self._fetch(sen_rnn, sen_lengths, batch_size, self.sen_len)  # (batch_size, hid)

        representation = sentence_batch
        out = self.output(representation)
        #out_prob = F.softmax(out.view(batch_size, self.output_dim), dim=1)
        return out
            
class BiLSTM(nn.Module):
    def __init__(self, embeddings, input_dim, vocab_size, hidden_dim, output_dim, num_layers, max_len, dropout):
        super(BiLSTM, self).__init__()
        self.emb = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=input_dim,
            padding_idx=0)
        if embeddings is not None:
            self.emb.weight = nn.Parameter(embeddings)
            self.emb.weight.requires_grad = True
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.sen_len = max_len
        
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        self.output = nn.Linear(2 * hidden_dim, output_dim)
        
    def bi_fetch(self, rnn_outs, seq_lens, batch_size, max_len):
        # (batch_size, max_len, 2 , -1) : 2 代表 forward and backward
        rnn_outs = rnn_outs.view(batch_size, max_len, 2, -1)
        # (batch_size, max_len, 1, -1)
        # 取出第二维(forward and backward)的张量
        fw_out = torch.index_select(rnn_outs, 2, Variable(torch.LongTensor([0])))
        fw_out = fw_out.view(batch_size * max_len, -1)
        
        bw_out = torch.index_select(rnn_outs, 2, Variable(torch.LongTensor([1])))
        bw_out = bw_out.view(batch_size * max_len, -1)

        batch_range = Variable(torch.LongTensor(range(batch_size))) * max_len
        batch_zeros = Variable(torch.zeros(batch_size).long())

        # 获取最后一个有意义时刻的输出的隐层向量
        fw_index = batch_range + seq_lens.view(batch_size) - 1
        fw_out = torch.index_select(fw_out, 0, fw_index)  # (batch_size, hid)

        bw_index = batch_range + batch_zeros
        bw_out = torch.index_select(bw_out, 0, bw_index)

        outs = torch.cat([fw_out, bw_out], dim=1)
        return outs
    
    def forward(self, sen_batch, sen_lengths, sen_mask_matrix):
        """
        :param sen_batch: (batch, sen_length), tensor for sentence sequence
        :param sen_lengths:
        :param sen_mask_matrix:
        :return:
        """
        ''' Embedding Layer | Padding | Sequence_length 40'''
        sen_batch = self.emb(sen_batch)
        batch_size = len(sen_batch)
        ''' Bi-LSTM Computation '''
        # output of shape (seq_len, batch, num_directions * hidden_size)
        # output: 最后一层每个timestamp的输出
        sen_outs, _ = self.rnn(sen_batch.view(batch_size, -1, self.input_dim))
        sen_rnn = sen_outs.contiguous().view(batch_size, -1, 2 * self.hidden_dim)  # (batch, sen_len, 2*hid)
        ''' Fetch the truly last hidden layer of both sides
        '''
        sentence_batch = self.bi_fetch(sen_rnn, sen_lengths, batch_size, self.sen_len)  # (batch_size, 2*hid)

        representation = sentence_batch
        out = self.output(representation)
        #out_prob = F.softmax(out.view(batch_size, self.output_dim), dim=1)
        return out
                
        
class RNNModel(nn.Module):
    """
    ntoken: dictionary size
    ninp: input vector dim
    nhid: hidden vector dim
    nlayers: layer num of NN
    """
    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)
        
        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight
        
        self.init_weights()
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        
    def init_weights(self):
        init_range = 0.1
        self.encoder.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)
        
    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        return decoded, hidden
    
    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)
        

class App(object):
    def __init__(self, train_file, test_file, model_name, pre_train_emb_name):
        self.used_pre_train_emb = False
        self.pre_train_emb_name = pre_train_emb_name
        self.model_name = model_name
        
        self.train_data = pd.read_csv(train_file, sep='\t', header=0)
        self.test_data = pd.read_csv(test_file, sep='\t', header=0)
        self.get_parameter()
        
    def get_model(self):
        models = {
            "LSTM" : LSTM_Model,
            "BILSTM" : BiLSTM,
        }
        model = models[self.model_name](
            embeddings=self.pre_train_emb_weight,
            input_dim=self.input_dim,
            vocab_size=self.vocab_size,
            hidden_dim=self.hidden_dim,
            num_layers=1,
            output_dim=self.output_dim,
            max_len=self.max_len, 
            dropout=self.dropout
        )
        return model
    
    def get_parameter(self):
        self.batch_size = 128
        self.hidden_dim = 128
        self.dropout = 0
        self.max_len = 50
        self.output_dim = 5
        self.lr = 0.001
        self.valid_part = 0.2
        self.epoch = 10
    
    def get_pre_train_emb(self, pre_train_emb_file):
        with open(pre_train_emb_file) as f:
            embeddings = []
            word2idx = {}
            word2idx["_padding"] = 0
            word2idx["_unk"] = 1
            
            for line in f:
                line = line.strip()
                parts = line.split(' ')
                word = parts[0]
                emb = [float(i) for i in parts[1:]]
                embeddings.append(emb)
                if word not in word2idx:
                    word2idx[word] = len(word2idx)
            
            ''' Add padding and unknown word to embeddings and word2idx'''
            emb_dim = len(embeddings[0])
            embeddings.insert(0, np.zeros(emb_dim))  # _padding
            embeddings.insert(1, np.random.random(emb_dim))  # _unk
            embeddings = np.asarray(embeddings, dtype=np.float32)
            embeddings = embeddings.reshape(len(embeddings), emb_dim)
            
            idx2word = dict((word2idx[word], word) for word in word2idx)
            self.vocab = {
                "embeddings" : embeddings,
                "word2idx" : word2idx,
                "idx2word" : idx2word,
            }
                
            #self.pre_train_emb_model = gensim.models.KeyedVectors.load_word2vec_format()
            self.pre_train_emb_weight = torch.from_numpy(embeddings)
            self.vocab_size, self.input_dim = self.pre_train_emb_weight.shape
    
    def test_prf(self, pred, labels):
        total = len(labels)
        pred_right  =    [0] * self.output_dim
        pred_all    =    [0] * self.output_dim
        gold        =    [0] * self.output_dim
        for i in range(total):
            pred_all[pred[i]] += 1
            if pred[i] == labels[i]:
                pred_right[pred[i]] += 1
            gold[labels[i]] += 1
        print(" Prediction:" + str(pred_all) + " Right:" + str(pred_right) + " Gold:" + str(gold))
        acc = 1.0 * sum(pred_right) / total
        p, r, f1 = Util.cal_prf(
            pred_all,
            pred_right,
            gold
        )
        _, _, macro_f1 = Util.cal_prf(
            pred_all,
            pred_right,
            gold,
            formation=False,
            metrics_type="macro"
        )
        print("acc on test is %d/%d = %f" % (sum(pred_right), total, acc), flush=True)
        print("Precision: %s\n  Recall: %s\n F1 score: %s\n Macro F1 score on test (Neg|Neu|Pos) is %s" % (str(p), str(r), str(f1), str(macro_f1)))
        return acc
         
    def test(self, model, test_X, test_X_seq_len, test_X_mask, test_Y=None):
        sentences_, sentences_seqlen_, sentences_mask_ = Util.var_batch(
            len(test_X),
            test_X,
            test_X_seq_len,
            test_X_mask,
            self.max_len
        )
        probs = model(sentences_, sentences_seqlen_, sentences_mask_)
        _, pred = torch.max(probs, dim=1)
        pred = pred.view(-1).data.numpy()
        if test_Y is not None:
            test_Y = np.asarray(test_Y)
            acc = self.test_prf(pred, test_Y)
        return pred
        
    def train(self, model, train_X, train_X_seq_len, train_X_mask, train_Y, optimizer, criterion):
        model.train()    
        sentences_, sentences_seqlen_, sentences_mask_ = Util.var_batch(
            self.batch_size,
            train_X,
            train_X_seq_len,
            train_X_mask,
            self.max_len
        )
        labels_ = Variable(torch.LongTensor(train_Y))
        model.zero_grad()
        probs = model(sentences_, sentences_seqlen_, sentences_mask_)
        loss = criterion(probs.view(len(labels_), -1), labels_)
        loss.backward()
        clip = 5
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
    
    def sentence_proc(self, sentence, stop_words, stemmer):
        #tokens = word_tokenize(sentence)
        tokens = sentence.split(" ")
        res_words = tokens
        res_words = [word.lower() for word in res_words]
        res_words = [ word for word in tokens if word not in stop_words]
        res_words = [stemmer.stem(word) for word in res_words]
        return res_words

    def text_process(self):
        stop_words = set(stopwords.words('english'))
        stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])
        stemmer = SnowballStemmer('english')
        
        raw_doc_train = self.train_data['Phrase'].values
        raw_doc_test = self.test_data['Phrase'].values
        processed_train_label = self.train_data['Sentiment'].values
        processed_train = []
        index = 0
        
        word2idx = {}
        word2idx["_padding"] = 0
        word2idx["_unk"] = 1
        for sentence in raw_doc_train:
            res_words = self.sentence_proc(sentence)
            processed_train.append(res_words)
            index += 1
            if self.used_pre_train_emb is False:
                for word in res_words:
                    if word not in word2idx:
                        word2idx[word] = len(word2idx)
        
        processed_test = []
        for sentence in raw_doc_test:
            res_words = self.sentence_proc(sentence)
            processed_test.append(res_words)
            if self.used_pre_train_emb is False:
                for word in res_words:
                    if word not in word2idx:
                        word2idx[word] = len(word2idx)
        
        if self.used_pre_train_emb is False:
            idx2word = dict((word2idx[word], word) for word in word2idx)
            self.vocab = {
                "embeddings" : None,
                "word2idx" : word2idx,
                "idx2word" : idx2word,
            }
            self.pre_train_emb_weight = None
            self.vocab_size, self.input_dim = len(word2idx), 128
            
        return processed_train, processed_test, processed_train_label
    
    def train_valid_set_split_static(self, data_X, data_X_len, data_Y):
        train_num = int(len(data_X) * (1 - self.valid_part))
        train_X = data_X[0:train_num]
        valid_X = data_X[train_num:]
        train_Y = data_Y[0:train_num]
        valid_Y = data_Y[train_num:]
        train_X_seq_len = data_X_len[0:train_num]
        valid_X_seq_len = data_X_len[train_num:]
        return train_X, train_X_seq_len, train_Y, valid_X, valid_X_seq_len, valid_Y
    
    def train_valid_set_split(self, data_X, data_X_len, data_Y):
        idxs = list(range(len(data_X)))
        random.shuffle(idxs)
        train_X, train_Y, train_X_seq_len = [], [], []
        valid_X, valid_Y, valid_X_seq_len = [], [], []
        train_num = int(len(data_X) * (1 - self.valid_part))
        for i in range(len(data_X)):
            idx = idxs[i]
            if i < train_num:
                train_X.append(data_X[idx])
                train_Y.append(data_Y[idx])
                train_X_seq_len.append(data_X_len[idx])
            else:
                valid_X.append(data_X[idx])
                valid_Y.append(data_Y[idx])
                valid_X_seq_len.append(data_X_len[idx])
        return train_X, train_X_seq_len, train_Y, valid_X, valid_X_seq_len, valid_Y
    
    def sentence_to_idx(self, data_X):
        data_idx = []
        for sentence in data_X:
            sen_idx = []
            for word in sentence:
                if word in self.vocab["word2idx"]:
                    sen_idx.append(self.vocab["word2idx"][word])
                else:
                    sen_idx.append(self.vocab["word2idx"]["_unk"])
                    print("unknown\t" + word)
            if all(v == 0 for v in sen_idx):
                print("ERROR: " + str(sentence))
            data_idx.append(sen_idx)
        return data_idx
    
    def sentence_pad(self, data_X):
        return Util.get_padding(data_X, self.max_len)

    def prepare_data(self):
        processed_train, processed_test, processed_train_label = self.text_process()
 
        print("1. sentence to idx")
        train_X_idx = self.sentence_to_idx(processed_train)
        test_X_idx = self.sentence_to_idx(processed_test)
        
        print("2. sentence padding")
        train_X_padded, train_X_seq_len = self.sentence_pad(train_X_idx)
        test_X_padded, test_X_seq_len = self.sentence_pad(test_X_idx)
        return train_X_padded, train_X_seq_len, test_X_padded, test_X_seq_len, processed_train_label

    def save_res(self, test_pred):
        self.test_data['Sentiment'] = test_pred.reshape(-1,1) 
        header = ['PhraseId', 'Sentiment']
        self.test_data.to_csv('./lstm_my_sentiment.csv', columns=header, index=False, header=True)

        
    def process(self):
        if self.used_pre_train_emb:
            self.get_pre_train_emb(self.pre_train_emb_name)
        
        best_acc_test, best_acc_valid = -np.inf, -np.inf
        
        train_X_padded, train_X_seq_len_padded, test_X_padded, test_X_seq_len_padded, train_Y_padded = self.prepare_data()
        
        model = self.get_model()
        '''
        optimizer = optim.SGD(
            model.parameters(),
            lr=self.lr,
            weight_decay=1e-5)
        '''
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.lr,
            weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        
        for i in range(self.epoch):
            print("epoch " + str(i))
            train_X, train_X_seq_len, train_Y, valid_X, valid_X_seq_len, valid_Y = self.train_valid_set_split(train_X_padded, train_X_seq_len_padded, train_Y_padded)
            
            batch_num = int(len(train_X) / self.batch_size)
            for j in range(batch_num):
                print("batch " + str(j), flush=True)
                start_idx = j * self.batch_size
                end_idx = (j + 1) * self.batch_size
                train_X_batch = train_X[start_idx : end_idx]
                train_Y_batch = train_Y[start_idx : end_idx]
                train_X_seq_len_batch = train_X_seq_len[start_idx : end_idx]
                
                self.train(model, train_X_batch, train_X_seq_len_batch, None, train_Y_batch, optimizer, criterion)
            print("test for epoch " + str(i))
            acc_score = self.test(model, valid_X, valid_X_seq_len, None, valid_Y)
        pred = self.test(model, test_X_padded, test_X_seq_len_padded, None, None)
        self.save_res(pred)
        

if __name__ == "__main__":
    dataset = "./dataset/"
    train_file = dataset + "train.tsv"
    test_file = dataset + "test.tsv"
    pre_train_model_path = "/Users/sniper/data/glove.6B.100d.txt"
    app = App(train_file, test_file, "BILSTM", pre_train_model_path)
    app.process()
    
    
