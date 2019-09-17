import numpy as np
import pandas as pd

from gensim import corpora
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader


df_train = pd.read_csv("./dataset/train.tsv", sep="\t")
df_test = pd.read_csv("./dataset/test.tsv", sep="\t")


def sentence_proc(sentence):
    tokens = sentence.split(" ")
    res_words = tokens
    res_words = [word.lower() for word in res_words]
    return res_words


def text_process():
    raw_doc_train = df_train['Phrase'].values
    raw_doc_test = df_test['Phrase'].values
    processed_train_label = df_train['Sentiment'].values
    processed_train = []

    word2idx = {}
    word2idx["_padding"] = 0
    #word2idx["_unk"] = 1
    for sentence in raw_doc_train:
        res_words = sentence_proc(sentence)
        processed_train.append(res_words)
        for word in res_words:
            if word not in word2idx:
                word2idx[word] = len(word2idx)

    processed_test = []
    for sentence in raw_doc_test:
        res_words = sentence_proc(sentence)
        processed_test.append(res_words)
        for word in res_words:
            if word not in word2idx:
                word2idx[word] = len(word2idx)

    idx2word = dict((word2idx[word], word) for word in word2idx)
    vocab = {
        "embeddings": None,
        "word2idx": word2idx,
        "idx2word": idx2word,
        "vocab_size": len(word2idx),
        "pre_train_emb_weight": None
    }
    return processed_train, processed_test, processed_train_label, vocab


processed_train, processed_test, processed_train_label, vocab = text_process()


def sentence_to_idx(data_X, vocab):
    data_idx = []
    for sentence in data_X:
        sen_idx = []
        for word in sentence:
            if word in vocab["word2idx"]:
                sen_idx.append(vocab["word2idx"][word])
            else:
                # sen_idx.append(vocab["word2idx"]["_unk"])
                print("unknown\t" + word)
        if all(v == 0 for v in sen_idx):
            print("ERROR: " + str(sentence))
        data_idx.append(sen_idx)
    return data_idx


def get_padding(sents, max_len):
    padded = []
    seq_len = []
    for sent in sents:
        num_words = len(sent)
        num_pad = max_len - num_words
        sent = np.asarray(sent[: max_len], dtype=np.int64).reshape(1, -1)
        if num_pad > 0:
            zero_paddings = np.zeros((1, num_pad), dtype=np.int64)
            sent = np.concatenate((sent, zero_paddings), axis=1)
        else:
            num_words = max_len
        padded.append(sent)
        seq_len.append(num_words)

    sent_batch_size = len(padded)
    padded = np.asarray(padded).reshape(sent_batch_size, max_len)
    seq_len = np.asarray(seq_len)
    return padded.astype(np.int64), seq_len.astype(np.int64)


max_len = 50


def prepare_data():
    train_X_idx = sentence_to_idx(processed_train, vocab)
    test_X_idx = sentence_to_idx(processed_test, vocab)
    train_X_padded, train_X_seq_len = get_padding(train_X_idx, max_len)
    test_X_padded, test_X_seq_len = get_padding(test_X_idx, max_len)
    return train_X_padded, train_X_seq_len, test_X_padded, test_X_seq_len


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
        output = self.fc(lstm_output[:, -1, :])
        out_prob = F.softmax(output.view(batch_size, self.output_dim), dim=1)
        return out_prob


class SentimentRNN(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """

    def __init__(self, embeddings, input_dim, vocab_size, hidden_dim, num_layers, output_dim, max_len=100, dropout=0.5):
        # def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        """
        Initialize the model by setting up the layers.
        """
        super(SentimentRNN, self).__init__()

        self.output_size = output_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # embedding and LSTM layers
        self.embedding = nn.Embedding(
            vocab_size,
            input_dim
        )
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            dropout=dropout,
            bidirectional=False,
            batch_first=True
        )

        # dropout layer
        self.dropout = nn.Dropout(0.3)

        # linear
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        batch_size = x.size(0)

        # embeddings and lstm_out
        embeds = self.embedding(x)

        lstm_out, hidden = self.lstm(embeds)

        # transform lstm output to input size of linear layers
        lstm_out = lstm_out.transpose(0, 1)
        lstm_out = lstm_out[-1]

        out = self.dropout(lstm_out)
        out = self.fc(out)
        return out
        #out_prob = F.softmax(out.view(batch_size, self.output_size), dim=1)
        #return out_prob
    '''
    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        batch_size = x.size(0)

        # embeddings and lstm_out
        embeds = self.embedding(x)

        lstm_out, hidden = self.lstm(embeds, hidden)

        # transform lstm output to input size of linear layers
        lstm_out = lstm_out.transpose(0, 1)
        lstm_out = lstm_out[-1]

        out = self.dropout(lstm_out)
        out = self.fc(out)

        return out, hidden

    def init_hidden(self, batch_size):
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        return hidden

    '''


def train_valid_test_set_split_static(data_X, data_X_len, data_Y):
        # split data into training, validation, and test data (features and labels, x and y)
    split_idx = int(len(data_X)*0.8)
    train_x, remaining_x = data_X[:split_idx], data_X[split_idx:]
    train_y, remaining_y = data_Y[:split_idx], data_Y[split_idx:]
    train_x_len, remain_x_len = data_X_len[:split_idx], data_X_len[split_idx:]
    test_idx = int(len(remaining_x)*0.5)
    val_x, test_x = remaining_x[:test_idx], remaining_x[test_idx:]
    val_y, test_y = remaining_y[:test_idx], remaining_y[test_idx:]
    val_x_len, test_x_len = remain_x_len[:test_idx], remain_x_len[test_idx:]

    # print out the shapes of  resultant feature data
    print("\t\t\tFeature Shapes:")
    print("Train set: \t\t{}".format(train_x.shape),
          "\nValidation set: \t{}".format(val_x.shape),
          "\nTest set: \t\t{}".format(test_x.shape))
    return train_x, val_x, test_x, train_y, val_y, test_y, train_x_len, val_x_len, test_x_len


def get_model(vocab):
    model = SentimentRNN(
        embeddings=None,
        input_dim=256,
        vocab_size=vocab['vocab_size'],
        hidden_dim=128,
        num_layers=1,
        output_dim=5,
        max_len=50,
        dropout=0.0
    )
    return model


train_X_padded, train_X_seq_len, test_X_padded, test_X_seq_len = prepare_data()
model = get_model(vocab)
optimizer = optim.Adam(
    model.parameters(),
    lr=0.002,
    weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
batch_size = 128
counter = 0
print_every = 20
clip = 5  # gradient clipping
epochs = 3
for i in range(epochs):
    num = len(train_X_padded)
    train_X_use = train_X_padded[0:num]
    train_X_seq_len_use = train_X_seq_len[0:num]
    processed_train_label_use = processed_train_label[0:num]

    train_X, val_X, test_X, train_Y, val_Y, test_Y, train_X_seq_len, valid_X_seq_len, test_X_seq_len = train_valid_test_set_split_static(
        train_X_use, train_X_seq_len_use, processed_train_label_use)

    train_dataset = TensorDataset(
        torch.from_numpy(train_X), torch.from_numpy(train_Y))
    valid_dataset = TensorDataset(
        torch.from_numpy(val_X), torch.from_numpy(val_Y))
    test_dataset = TensorDataset(
        torch.from_numpy(test_X), torch.from_numpy(test_Y))

    # dataloaders
    #batch_size = 54

    # make sure the SHUFFLE your training data
    train_loader = DataLoader(
        train_dataset, shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(
        valid_dataset, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)

    # Check the size of the loaders (how many batches inside)
    print(len(train_loader))
    print(len(valid_loader))
    print(len(test_loader))

    for inputs, labels in train_loader:
        counter += 1

        model.zero_grad()
        output = model(inputs)
        loss = criterion(output, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        if counter % print_every == 0:
            val_losses = []
            model.eval()
            for inputs, labels in valid_loader:
                output = model(inputs)
                val_loss = criterion(output, labels)
                val_losses.append(val_loss.item())
            model.train()
            print("Epoch: {}/{}...".format(i+1, epochs),
                    "Step: {}...".format(counter),
                    "Loss: {:.6f}...".format(loss.item()),
                    "Val Loss: {:.6f}".format(np.mean(val_losses))
            )
        
test_losses = []
num_correct = 0

model.eval()
for inputs, labels in test_loader:
    output = model(inputs)
    test_loss = criterion(output, labels)
    test_losses.append(test_loss.item())
    _, pred = torch.max(output,1)
    correct_tensor = pred.eq(labels.view_as(pred))
    correct = np.squeeze(correct_tensor.numpy())
    num_correct += np.sum(correct)
    
print("Test loss: {:.3f}".format(np.mean(test_losses)))

# accuracy over all test data
test_acc = num_correct/len(test_loader.dataset)
print("Test accuracy: {:.3f}".format(test_acc))